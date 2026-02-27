% --- Experimental Parameters ---
kValues = [1];                  
FramesPerRouter = [300];    
SNRList = [-5];           

originalNumKnownRouters = 67;
originalNumUnknownRouters = 20;

startSNR = -5;
startFramesPerRouter = FramesPerRouter(1);
startK = kValues(1);
startProcessing = true;

rng(123456);
enableParallel = true;     
showTrainingPlot = true;   
useLegacyAlpha = false;       

if exist('LabelSmoothingClassificationLayer','class') ~= 8
    try
        addpath(fileparts(mfilename('fullpath')));
        rehash;
    catch
    end
end

baseCheckpointDir = fullfile(pwd, 'model_checkpoints');
if ~exist(baseCheckpointDir, 'dir')
    mkdir(baseCheckpointDir);
end

if enableParallel && isempty(gcp('nocreate'))
    try
        parpool;
    catch
    end
end

% --- Main Loop ---
for localSNR = SNRList
    for localFramesPerRouter = FramesPerRouter
        for k = kValues
            if ~startProcessing
                if localSNR == startSNR && localFramesPerRouter == startFramesPerRouter && k == startK
                    startProcessing = true;
                else
                    continue;
                end
            end

            fprintf('\n======================================================\n');
            fprintf('Processing SNR = %d dB\n', localSNR);
            fprintf('======================================================\n');

            numKnownRouters = originalNumKnownRouters * k;
            numUnknownRouters = originalNumUnknownRouters * k;
            numTotalRouters = numKnownRouters + numUnknownRouters;

            SNR = localSNR;           
            channelNumber = 1;
            channelBand = 5;
            frameLength = 160;
            san = 0.5;                

            numTotalFramesPerRouter = localFramesPerRouter; 
            numTrainingFramesPerRouter = floor(numTotalFramesPerRouter*0.8);
            numValidationFramesPerRouter = floor(numTotalFramesPerRouter*0.1);
            numTestFramesPerRouter = numTotalFramesPerRouter - numTrainingFramesPerRouter - numValidationFramesPerRouter;

            all_alpha = zeros(1,numTotalRouters);
            all_beta = zeros(1,numTotalRouters);
            if useLegacyAlpha
                alphaSampler = @generateAlphaLegacy;
            else
                alphaSampler = @generateAlpha;
            end
            for idx = 1:numTotalRouters
                alpha = alphaSampler(san);
                beta = (alpha - 1) + 0.2 * rand(1) - 0.1;
                all_alpha(idx)= alpha;
                all_beta(idx)= beta;
            end

            fc = wlanChannelFrequency(channelNumber, channelBand);
            radioImpairments = repmat(struct('PhaseNoise', 0, 'DCOffset', 0, 'FrequencyOffset', 0), numTotalRouters, 1);
            for routerIdx = 1:numTotalRouters
                radioImpairments(routerIdx).PhaseNoise = rand*(0.29) + 0.01;
                radioImpairments(routerIdx).DCOffset = rand*(18) - 50;
                radioImpairments(routerIdx).FrequencyOffset = fc/1e6*(rand*(8) - 4);
            end

            rawTrainData = cell(1, numTotalRouters);
            rawValData = cell(1, numTotalRouters);
            rawTestData = cell(1, numTotalRouters);
            idx_train = 1:numTrainingFramesPerRouter;
            idx_val = numTrainingFramesPerRouter + (1:numValidationFramesPerRouter);
            idx_test = numTrainingFramesPerRouter + numValidationFramesPerRouter + (1:numTestFramesPerRouter);

            tic;
            generatedMACAddresses = strings(numTotalRouters, 1);
            routerIndices = 1:numTotalRouters; 
            D = parallel.pool.DataQueue;
            afterEach(D, @(x) updateProgress(x, numTotalRouters));
            fprintf('Step 1/4: Generating wireless signals...\nProgress: 0.0%%\n');

            parfor idx = 1:numTotalRouters
                routerIdx = routerIndices(idx);  
                if routerIdx <= numKnownRouters
                    currentMAC = string(dec2hex(bi2de(randi([0 1], 12, 4)))');
                else
                    currentMAC = 'AAAAAAAAAAAA';
                end
                generatedMACAddresses(idx) = currentMAC; 

                loc_frameBody = wlanMACManagementConfig;
                loc_beaconConfig = wlanMACFrameConfig('FrameType', 'Beacon', "ManagementConfig", loc_frameBody);
                loc_beaconConfig.Address2 = currentMAC;

                [~, loc_mpduLength] = wlanMACFrame(loc_beaconConfig, 'OutputFormat', 'bits');
                loc_nonHTConfig = wlanNonHTConfig('ChannelBandwidth', "CBW20", "MCS", 1, "PSDULength", loc_mpduLength);
                loc_fs = wlanSampleRate(loc_nonHTConfig);

                loc_rxFrontEnd = rfFingerprintingNonHTFrontEnd('ChannelBandwidth', 'CBW20');
                loc_multipathChannel = comm.RayleighChannel('SampleRate', loc_fs, ...
                    'PathDelays', [0 1.8 3.4]/loc_fs, ...
                    'AveragePathGains', [0 -2 -10], ...
                    'MaximumDopplerShift', 0);

                beacon = wlanMACFrame(loc_beaconConfig, 'OutputFormat', 'bits');
                txWaveform = wlanWaveformGenerator(beacon, loc_nonHTConfig);
                txWaveform = helperNormalizeFramePower(txWaveform);
                txWaveform = [txWaveform; zeros(160,1)]; 

                frameCount = 0;
                trials = 0;
                maxTrials = max(5*numTotalFramesPerRouter, 200);
                local_rxLLTF = zeros(frameLength, numTotalFramesPerRouter, 'single'); 
                currentAlpha = all_alpha(idx);
                currentBeta = all_beta(idx);
                currentImp = radioImpairments(idx);

                while frameCount < numTotalFramesPerRouter && trials < maxTrials  
                    trials = trials + 1; 
                    rxMultipath = loc_multipathChannel(txWaveform); 
                    rxImpairment = helperRFImpairments(rxMultipath, currentImp, loc_fs); 
                    rxSigFE = rxImpairment;

                    [valid, ~, ~, ~, ~, LLTF] = loc_rxFrontEnd(rxSigFE); 
                    if valid  
                        LLTF = single(LLTF); 
                        LLTF = LLTF .* LLTF .* currentAlpha ./ (1 + currentBeta * LLTF .* LLTF);

                        % Add AWGN based on specified SNR
                        if SNR <= 0
                            Ps = mean(abs(LLTF).^2 + eps);
                            Nvar = Ps / 10^(SNR/10);
                            noise = sqrt(Nvar/2) * (randn(size(LLTF)) + 1j*randn(size(LLTF)));
                            LLTF = LLTF + noise;
                        end

                        frameCount = frameCount + 1; 
                        local_rxLLTF(:, frameCount) = LLTF;
                    end
                end

                if frameCount < numTotalFramesPerRouter
                    if frameCount > 0
                        local_rxLLTF(:, frameCount+1:end) = repmat(local_rxLLTF(:,frameCount), 1, numTotalFramesPerRouter-frameCount);
                    else
                        local_rxLLTF(:, :) = 0;
                    end
                end

                local_rxLLTF = local_rxLLTF(:, randperm(numTotalFramesPerRouter));
                rawTrainData{idx} = local_rxLLTF(:, idx_train);
                rawValData{idx}   = local_rxLLTF(:, idx_val);
                rawTestData{idx}  = local_rxLLTF(:, idx_test); 
                send(D, 1);
            end

            fprintf('\nSignal generation time: %.2f s\n', toc);

            xTrainingFrames = cat(2, rawTrainData{:});
            clear rawTrainData; 
            xValFrames = cat(2, rawValData{:});
            clear rawValData; 
            xTestFrames = cat(2, rawTestData{:});
            clear rawTestData;

            yTrain_all = strings(0);
            yVal_all = strings(0);
            yTest_all = strings(0);
            for i = 1:numTotalRouters
                currLabel = generatedMACAddresses(i); 
                yTrain_all = [yTrain_all; repelem(currLabel, numTrainingFramesPerRouter, 1)]; 
                yVal_all   = [yVal_all;   repelem(currLabel, numValidationFramesPerRouter, 1)]; 
                yTest_all  = [yTest_all;  repelem(currLabel, numTestFramesPerRouter, 1)];
            end
            
            yTrain_all = categorical(yTrain_all); 
            yVal_all   = categorical(yVal_all,   categories(yTrain_all)); 
            yTest_all  = categorical(yTest_all,  categories(yTrain_all));
            numTrain_all = numel(yTrain_all);
            numVal_all = numel(yVal_all);
            numTest_all = numel(yTest_all);

            xTrainingFrames = reshape(cat(2, real(xTrainingFrames), imag(xTrainingFrames)), [frameLength, 2, numTrain_all]); 
            xValFrames      = reshape(cat(2, real(xValFrames),      imag(xValFrames)),      [frameLength, 2, numVal_all]); 
            xTestFrames     = reshape(cat(2, real(xTestFrames),     imag(xTestFrames)),     [frameLength, 2, numTest_all]);

            knownIdxTrain = (yTrain_all ~= "AAAAAAAAAAAA"); 
            yTrain = yTrain_all(knownIdxTrain); 
            xTrainingFrames = xTrainingFrames(:,:,knownIdxTrain);

            knownIdxVal = (yVal_all ~= "AAAAAAAAAAAA"); 
            yVal = yVal_all(knownIdxVal); 
            xValFrames = xValFrames(:,:,knownIdxVal);

            knownIdxTest = (yTest_all ~= "AAAAAAAAAAAA"); 
            yTest_Known = yTest_all(knownIdxTest); 
            xTestFrames_Known = xTestFrames(:,:,knownIdxTest);
            unknownIdxTest = (yTest_all == "AAAAAAAAAAAA"); 
            yTest_Unknown = yTest_all(unknownIdxTest); 
            xTestFrames_Unknown = xTestFrames(:,:,unknownIdxTest); %#ok<NASGU>

            numTrain = numel(yTrain); 
            vr = randperm(numTrain); 
            xTrainingFrames = xTrainingFrames(:,:,vr); 
            yTrain = yTrain(vr);

            allCats = categories(yTrain); 
            allCats(allCats == "AAAAAAAAAAAA") = []; 
            classNames = allCats; 
            yTrain = removecats(yTrain, "AAAAAAAAAAAA"); 
            yVal   = removecats(yVal,   "AAAAAAAAAAAA"); 
            numClasses = numel(classNames); 

            fprintf('Step 2/4: Computing global statistics...\n');
            subsetSize = min(2000, numTrain); 
            tempSubset = xTrainingFrames(:,:,1:subsetSize); 
            tempSWT = applySWTToBatch_IQ(tempSubset); 
            allTrainData = reshape(tempSWT, [], 4);   
            globalMu = median(allTrainData, 1, 'omitnan'); 
            globalSigma = mad(allTrainData, 1, 1) * 1.4826; 
            globalSigma(globalSigma < 1e-6) = 1; 
            clear allTrainData tempSubset tempSWT; 

            D_Aug = parallel.pool.DataQueue; 
            afterEach(D_Aug, @(x) updateProgress(x, numTrain));
            fprintf('Step 3/4: Constructing training set...\nProgress: 0.0%%\n');
            XTrain = cell(numTrain,1);
            
            parfor i = 1:numTrain 
                Xi = squeeze(xTrainingFrames(:, :, i)).'; 
                Xi = single(Xi); 
                Xi = normalizePerSequence(Xi); 
                Xi = augmentSequence_IQ(Xi, SNR); 
                Xi_SWT = computeSingleSWT(Xi); 
                Xi_SWT = (Xi_SWT.' - globalMu) ./ globalSigma; 
                XTrain{i} = Xi_SWT.'; 
                send(D_Aug, 1);
            end
            fprintf('\n'); 
            clear xTrainingFrames; 

            XVal = processEvalData(xValFrames, globalMu, globalSigma); 
            clear xValFrames;

            % --- Model Architecture Construction ---
            inputFeatureSize = 4; 
            embedDim = 512; 
            
            lgraph = layerGraph(); 
            [lgraph, inputName] = addInputLayer1D(lgraph, inputFeatureSize); 
            
            [lgraph, lastName] = addMultiScaleFrontend(lgraph, inputName, 64, 'ms1');
            [lgraph, lastName] = addMultiScaleFrontend(lgraph, lastName, 64, 'ms2');
            [lgraph, lastName] = addStemBlock(lgraph, lastName, 64);
            [lgraph, lastName] = addResNetBackbone1D(lgraph, lastName, embedDim);
            [lgraph, lastName] = addDilatedBlock(lgraph, lastName, embedDim, [2, 4], 'trans_res_attn');
            [lgraph, lastName] = addGlobalAttentionBlock(lgraph, lastName, embedDim);
            [lgraph, lastName] = addDilatedBlock(lgraph, lastName, embedDim, [1, 2, 4, 8], 'trans_attn_lstm');
            [lgraph, lastName] = addBiLSTMStackStrong(lgraph, lastName);
            
            radius = 16; 
            [lgraph, lastName] = addHypersphereHead(lgraph, lastName, numClasses, classNames, radius);

            % --- Training Configuration ---
            miniBatchSize = 128; 
            maxEpochs = 20;        

            if SNR == -5
                initialLR = 5e-5;
            elseif SNR == 0
                initialLR = 8e-5;
            else
                initialLR = 2e-4;
            end

            fprintf('Training Options: Batch=%d, MaxEpochs=%d, InitialLR=%.1e (SNR=%d dB)\n', ...
                miniBatchSize, maxEpochs, initialLR, SNR);

            options = trainingOptions('adam', ...
                'MaxEpochs', maxEpochs, ... 
                'ValidationData', {XVal, yVal}, ...
                'ValidationFrequency', 200, ...
                'Verbose', true, ...
                'InitialLearnRate', initialLR, ... 
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropFactor', 0.3, ...     
                'LearnRateDropPeriod', 15, ...      
                'GradientThreshold', 0.5, ...       
                'MiniBatchSize', miniBatchSize, ...
                'Shuffle', 'every-epoch', ...
                'Plots', 'training-progress', ...
                'ExecutionEnvironment', 'auto');

            tic;
            fprintf('Step 4/4: Training started...\n');
            [lastNet, trainInfo] = trainNetwork(XTrain, yTrain, lgraph, options); 
            TrainTime = toc; %#ok<NASGU>
            bestNet = lastNet;
            
            % --- Evaluation ---
            fprintf('Evaluating on test set (Known Routers)...\n');
            XTest_Known = processEvalData(xTestFrames_Known, globalMu, globalSigma);
            YPred_Known = classify(bestNet, XTest_Known, 'MiniBatchSize', miniBatchSize);
            accTestKnown = mean(YPred_Known == yTest_Known);
            fprintf('Test accuracy on known routers (SNR = %d dB): %.2f%% (N = %d)\n', ...
                localSNR, accTestKnown * 100, numel(yTest_Known));

            saveFileName = fullfile(baseCheckpointDir, sprintf('Model_SNR_%d_Epoch%d.mat', localSNR, maxEpochs));
            save(saveFileName, 'bestNet', 'trainInfo', 'globalMu', 'globalSigma', 'classNames');
            fprintf('Training complete. Model saved to: %s\n', saveFileName);
        end
    end
end

% =========================================================================
% Utility Functions
% =========================================================================

function Xi_SWT = computeSingleSWT(Xi_IQ)
    sig_I = Xi_IQ(1, :); 
    sig_Q = Xi_IQ(2, :); 
    hasWavelet = ~isempty(which('swt')); 
    if hasWavelet
        [ca_i, cd_i] = swt(sig_I, 1, 'haar'); 
        [ca_q, cd_q] = swt(sig_Q, 1, 'haar'); 
    else
        ca_i = (sig_I + circshift(sig_I,1))/2; 
        cd_i = (sig_I - circshift(sig_I,1))/2; 
        ca_q = (sig_Q + circshift(sig_Q,1))/2; 
        cd_q = (sig_Q - circshift(sig_Q,1))/2; 
    end
    Xi_SWT = [ca_i; cd_i; ca_q; cd_q]; 
end

function X_SWT = applySWTToBatch_IQ(frames_IQ)
    [L, ~, N] = size(frames_IQ); 
    X_SWT = zeros(L, 4, N); 
    hasWavelet = ~isempty(which('swt')); 
    for i = 1:N
        sig_I = frames_IQ(:, 1, i); 
        sig_Q = frames_IQ(:, 2, i); 
        if hasWavelet
            [ca_i, cd_i] = swt(sig_I, 1, 'haar'); 
            [ca_q, cd_q] = swt(sig_Q, 1, 'haar'); 
            X_SWT(:,:,i) = [ca_i(:), cd_i(:), ca_q(:), cd_q(:)]; 
        else
            ca_i = (sig_I + circshift(sig_I,1))/2; 
            cd_i = (sig_I - circshift(sig_I,1))/2; 
            ca_q = (sig_Q + circshift(sig_Q,1))/2; 
            cd_q = (sig_Q - circshift(sig_Q,1))/2; 
            X_SWT(:,:,i) = [ca_i, cd_i, ca_q, cd_q]; 
        end
    end
end

function X_Out = processEvalData(Data_IQ, gMu, gSig)
    N = size(Data_IQ, 3); 
    X_Out = cell(N, 1); 
    parfor j = 1:N
        Xi = squeeze(Data_IQ(:, :, j)).'; 
        Xi = normalizePerSequence(Xi); 
        Xi_SWT = computeSingleSWT(Xi); 
        Xi_SWT = (Xi_SWT.' - gMu) ./ gSig; 
        X_Out{j} = Xi_SWT.'; 
    end
end

function [lgraph, inputName] = addInputLayer1D(lgraph, inputFeatureSize)
    inputName = 'input'; 
    lgraph = addLayers(lgraph, ...
        sequenceInputLayer(inputFeatureSize, 'Name', inputName, 'Normalization', 'none')); 
end

function [lgraph, outName, outChannels] = addMultiScaleFrontend(lgraph, inName, numFusionFilters, blockName)
    fSizes = [3 5 7 1];
    for i = 1:4
        branchName = sprintf('%s_br%d', blockName, fSizes(i)); 
        layers = [
            convolution1dLayer(fSizes(i), numFusionFilters/4, 'Padding', 'same', 'Name', [branchName '_conv'])
            batchNormalizationLayer('Name', [branchName '_bn'])
            reluLayer('Name', [branchName '_relu'])
        ]; 
        lgraph = addLayers(lgraph, layers); 
        lgraph = connectLayers(lgraph, inName, [branchName '_conv']);
    end
    concatName = [blockName '_concat']; 
    concat = depthConcatenationLayer(4, 'Name', concatName); 
    lgraph = addLayers(lgraph, concat); 
    lgraph = connectLayers(lgraph, sprintf('%s_br3_relu', blockName), [concatName '/in1']); 
    lgraph = connectLayers(lgraph, sprintf('%s_br5_relu', blockName), [concatName '/in2']); 
    lgraph = connectLayers(lgraph, sprintf('%s_br7_relu', blockName), [concatName '/in3']); 
    lgraph = connectLayers(lgraph, sprintf('%s_br1_relu', blockName), [concatName '/in4']);

    outChannels = numFusionFilters; 
    fusion = [
        convolution1dLayer(1, outChannels, 'Padding', 'same', 'Name', [blockName '_fusion'])
        batchNormalizationLayer('Name', [blockName '_fusion_bn'])
    ]; 
    lgraph = addLayers(lgraph, fusion); 
    lgraph = connectLayers(lgraph, concatName, [blockName '_fusion']); 

    match = convolution1dLayer(1, outChannels, 'Padding', 'same', 'Name', [blockName '_match']); 
    lgraph = addLayers(lgraph, match); 
    lgraph = connectLayers(lgraph, inName, [blockName '_match']); 

    addL = additionLayer(2, 'Name', [blockName '_add']); 
    lgraph = addLayers(lgraph, addL); 
    lgraph = connectLayers(lgraph, [blockName '_fusion_bn'], [addL.Name '/in1']); 
    lgraph = connectLayers(lgraph, [blockName '_match'],      [addL.Name '/in2']); 

    outRelu = reluLayer('Name', [blockName '_out_relu']); 
    lgraph = addLayers(lgraph, outRelu); 
    lgraph = connectLayers(lgraph, [blockName '_add'], [blockName '_out_relu']); 
    outName = [blockName '_out_relu'];
end

function [lgraph, outName] = addStemBlock(lgraph, inName, numFilters)
    stemPrefix = 'stem'; 
    mainLayers = [
        convolution1dLayer(7, numFilters, 'Padding', 'same', 'Name', [stemPrefix '_conv'])
        batchNormalizationLayer('Name', [stemPrefix '_bn'])
        leakyReluLayer(0.1, 'Name', [stemPrefix '_lrelu'])
        dropoutLayer(0.2, 'Name', [stemPrefix '_drop'])
    ]; 
    lgraph = addLayers(lgraph, mainLayers); 
    lgraph = connectLayers(lgraph, inName, [stemPrefix '_conv']); 
    outName = [stemPrefix '_drop']; 
end

function [lgraph, outName] = addResNetBackbone1D(lgraph, inName, embedDim)
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res1', 64,   64,   1, inName); 
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res2', 64,   128,  2, outName); 
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res3', 128,  128,  1, outName); 
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res4', 128,  embedDim, 2, outName); 
    [lgraph, outName] = addResidualBlock1D(lgraph, 'res5', embedDim, embedDim, 1, outName); 
end

function [lgraph, outName] = addResidualBlock1D(lgraph, blockName, inChannels, outChannels, stride, inputName)
    mainLayers = [
        convolution1dLayer(3, outChannels, 'Padding', 'same', 'Stride', stride, 'Name', blockName + "_conv1")
        batchNormalizationLayer('Name', blockName + "_bn1")
        reluLayer('Name', blockName + "_relu1")
        dropoutLayer(0.05, 'Name', blockName + "_drop1")
        convolution1dLayer(3, outChannels, 'Padding', 'same', 'Stride', 1, 'Name', blockName + "_conv2")
        batchNormalizationLayer('Name', blockName + "_bn2")
    ]; 
    skipNeeded = (inChannels ~= outChannels) || (stride ~= 1); 
    if skipNeeded
        skipLayers = [
            convolution1dLayer(1, outChannels, 'Padding', 'same', 'Stride', stride, 'Name', blockName + "_skip_conv")
            batchNormalizationLayer('Name', blockName + "_skip_bn")
        ]; 
        lgraph = addLayers(lgraph, skipLayers); 
    end
    lgraph = addLayers(lgraph, mainLayers); 
    addL = additionLayer(2, 'Name', blockName + "_add"); 
    lgraph = addLayers(lgraph, addL); 
    reluOut = reluLayer('Name', blockName + "_out"); 
    lgraph = addLayers(lgraph, reluOut); 

    lgraph = connectLayers(lgraph, inputName, blockName + "_conv1"); 
    if skipNeeded
        lgraph = connectLayers(lgraph, inputName, blockName + "_skip_conv"); 
        lgraph = connectLayers(lgraph, blockName + "_skip_bn", blockName + "_add/in2"); 
    else
        lgraph = connectLayers(lgraph, inputName, blockName + "_add/in2"); 
    end
    lgraph = connectLayers(lgraph, blockName + "_bn2",  blockName + "_add/in1"); 
    lgraph = connectLayers(lgraph, blockName + "_add",  blockName + "_out"); 
    outName = blockName + "_out"; 
end

function [lgraph, outName] = addDilatedBlock(lgraph, inName, numFilters, dilations, blockPrefix)
    lastLayerName = inName; 
    for i = 1:length(dilations)
        d = dilations(i); 
        convName = sprintf('%s_dil_conv_%d', blockPrefix, i); 
        bnName   = sprintf('%s_dil_bn_%d',   blockPrefix, i); 
        reluName = sprintf('%s_dil_relu_%d', blockPrefix, i); 
        layers = [
            convolution1dLayer(3, numFilters, 'Padding', 'same', 'DilationFactor', d, 'Name', convName)
            batchNormalizationLayer('Name', bnName)
            reluLayer('Name', reluName)
        ]; 
        lgraph = addLayers(lgraph, layers); 
        lgraph = connectLayers(lgraph, lastLayerName, convName); 
        lastLayerName = reluName; 
    end
    outName = lastLayerName; 
end

function [lgraph, outName] = addGlobalAttentionBlock(lgraph, inName, dim)
    attn = selfAttentionLayer(8, dim/8, 'Name', 'global_attn', 'DropoutProbability', 0.1); 
    lgraph = addLayers(lgraph, attn); 
    lgraph = connectLayers(lgraph, inName, 'global_attn'); 

    addL = additionLayer(2, 'Name', 'attn_add'); 
    lgraph = addLayers(lgraph, addL); 
    lgraph = connectLayers(lgraph, inName,      'attn_add/in1'); 
    lgraph = connectLayers(lgraph, 'global_attn','attn_add/in2'); 

    bn = batchNormalizationLayer('Name', 'attn_bn'); 
    lgraph = addLayers(lgraph, bn); 
    lgraph = connectLayers(lgraph, 'attn_add', 'attn_bn'); 
    outName = 'attn_bn'; 
end

function [lgraph, outName] = addBiLSTMStackStrong(lgraph, inName)
    bilstmLayers = [
        bilstmLayer(384, 'OutputMode', 'sequence', 'Name', 'bilstm1') 
        dropoutLayer(0.3, 'Name', 'rnn_drop1')
        bilstmLayer(256, 'OutputMode', 'sequence', 'Name', 'bilstm2') 
        dropoutLayer(0.3, 'Name', 'rnn_drop2')
    ];
    lgraph = addLayers(lgraph, bilstmLayers);
    lgraph = connectLayers(lgraph, inName, 'bilstm1');

    avgPool = globalAveragePooling1dLayer('Name', 'gap');
    lgraph = addLayers(lgraph, avgPool);
    lgraph = connectLayers(lgraph, 'rnn_drop2', 'gap');

    maxPool = globalMaxPooling1dLayer('Name', 'gmp');
    lgraph = addLayers(lgraph, maxPool);
    lgraph = connectLayers(lgraph, 'rnn_drop2', 'gmp');

    concat = depthConcatenationLayer(2, 'Name', 'pool_concat');
    lgraph = addLayers(lgraph, concat);
    lgraph = connectLayers(lgraph, 'gap', 'pool_concat/in1');
    lgraph = connectLayers(lgraph, 'gmp', 'pool_concat/in2');
    outName = 'pool_concat';
end

function [lgraph, outName] = addHypersphereHead(lgraph, inName, numClasses, classNames, radius)
    layers = [
        fullyConnectedLayer(256, 'Name', 'fc_bottleneck')
        reluLayer('Name', 'relu_bottleneck')
        dropoutLayer(0.35, 'Name', 'drop_bottleneck')
        l2NormalizationLayer('Name', 'l2_norm')
        scalingLayer('Name', 'feature_vector', 'Scale', radius)
        fullyConnectedLayer(numClasses, 'Name', 'fc_angle')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output', 'Classes', classNames)
    ];
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, inName, 'fc_bottleneck');
    outName = 'output';
end

function Z = augmentSequence_IQ(Z, SNR)
    [~, T] = size(Z);
    if SNR <= 0 
        pM = 0.25;   
        pG = 0.10;   
        pF = 0.15;   
        pN = 0.25;   
        nStd = 0.03;
    else 
        pM = 0.15;
        pG = 0.20;
        pF = 0.15;
        pN = 0.15;
        nStd = 0.02; 
    end

    s = randi([-5, 5]); 
    if s ~= 0
        Z = circshift(Z, [0 s]);
    end

    if rand < 0.5
        Z(2, :) = -Z(2, :);
    end

    theta = 2 * pi * rand(); 
    RotMatrix = [cos(theta) -sin(theta); sin(theta) cos(theta)]; 
    Z(1:2, :) = RotMatrix * Z(1:2, :);

    if rand < pM && T > 24
        ml = randi([8,22]); 
        t0 = randi([1, max(1, T-ml+1)]); 
        Z(:, t0:min(T, t0+ml-1)) = 0; 
    end

    if rand < pG
        Z(1:2,:) = Z(1:2,:) * 10^(randn*0.02); 
    end

    if rand < pF
        for ch=1:2
            x = Z(ch,:); 
            Xf = fft(x); 
            m = abs(Xf); 
            sm = sort(m,'descend'); 
            th = sm(max(1,round(0.25*numel(sm))))*0.15; 
            ms = ones(size(m)); 
            ms(m<th) = 0.8;      
            Z(ch,:) = real(ifft(Xf.*ms)); 
        end
    end

    if rand < pN
        Z = Z + nStd*randn(size(Z)); 
    end
end

function updateProgress(~, total)
    persistent count; 
    if isempty(count)
        count = 0;
    end 
    count = count + 1; 
    if mod(count, ceil(total/50)) == 0 || count == total
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\bProgress: %5.1f%%', (count/total)*100); 
    end
    if count == total
        count = [];
    end 
end

function [impairedSig] = helperRFImpairments(sig, radioImpairments, fs)
    fOff = comm.PhaseFrequencyOffset('FrequencyOffset', radioImpairments.FrequencyOffset,  'SampleRate', fs); 
    phaseNoise = helperGetPhaseNoise(radioImpairments); 
    phNoise = comm.PhaseNoise('Level', phaseNoise, 'FrequencyOffset', abs(radioImpairments.FrequencyOffset)); 
    impairedSig = phNoise(fOff(sig)) + 10^(radioImpairments.DCOffset/10); 
end

function y = helperNormalizeFramePower(x)
    y = x ./ sqrt(mean(abs(x).^2 + eps)); 
end

function [phaseNoise] = helperGetPhaseNoise(radioImpairments)
    persistent pMrms pMyI pxI pLoaded; 
    if isempty(pLoaded)
        try
            data = load('Mrms.mat','Mrms','MyI','xI'); 
            pMrms = data.Mrms; 
            pMyI = data.MyI; 
            pxI = data.xI; 
            pLoaded = true; 
        catch
            pLoaded = false; 
        end
    end
    if pLoaded
        [~, iRms] = min(abs(radioImpairments.PhaseNoise - pMrms)); 
        [~, iFreqOffset] = min(abs(pxI - abs(radioImpairments.FrequencyOffset))); 
        phaseNoise = -abs(pMyI(iRms, iFreqOffset)); 
    else
        phaseNoise = -80; 
    end
end

function alpha = generateAlpha(san)
    alpha = min(max(1 + san*randn(),0.7),1.3); 
end

function alpha = generateAlphaLegacy(san)
    mu = 1.5;
    sigma = san; 
    alpha = mu + sigma * randn(1, 1); 
    while alpha < 1.2 || alpha > 2.8
        alpha = mu + sigma * randn(1, 1); 
    end
end

function Z = normalizePerSequence(Z)
    for r = 1:size(Z,1)
        Z(r,:) = (Z(r,:) - mean(Z(r,:))) / sqrt(mean((Z(r,:) - mean(Z(r,:))).^2) + 1e-8); 
    end
end