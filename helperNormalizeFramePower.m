function y = helperNormalizeFramePower(x)
% Normalize average power of frame to unity
    p = mean(abs(x).^2 + eps);
    y = x ./ sqrt(p);
end