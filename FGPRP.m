% fuzzy_prp_simple.m  
%------------------------------------------------------------------------------
% Compute a fuzzy-granulated Probabilistic Recurrence Plot (PRP) from 1D data
% (no local manifold smoothing, Z-shaped membership)
%------------------------------------------------------------------------------
% Usage:
%   PRP = fuzzy_prp_simple(x, tau, m, G, sigma_prp)
%
% Inputs:
%   x         - 1D signal (vector)
%   tau       - time delay for embedding
%   m         - embedding dimension
%   G         - number of fuzzy granules per dimension
%   sigma_prp - Gaussian kernel width for probabilistic RP
%
% Output:
%   PRP       - N-by-N probabilistic recurrence matrix
%------------------------------------------------------------------------------
function PRP = FGPRP(x, tau, m, G, sigma_prp)
    % Ensure column vector
    x = x(:);

    % Delay embedding
    N = length(x) - (m-1)*tau;
    Y = zeros(N, m);
    for i = 1:N
        idx = i : tau : i + (m-1)*tau;
        Y(i, :) = x(idx).';
    end

    % Fuzzy granulation: define granule centers uniformly over Y
    dataMin = min(Y(:));
    dataMax = max(Y(:));
    centers = linspace(dataMin, dataMax, G);
    granuleWidth = centers(2) - centers(1);

    % Build Z-shaped membership tensor: N x m x G
    Memb = zeros(N, m, G);
    for dim = 1:m
        for g = 1:G
            c = centers(g);
            a = c - 1.5*granuleWidth;   % 1→0 的起点
            b = c + 1.5*granuleWidth;   % 1→0 的终点
            % MATLAB 自带 zmf
            Memb(:, dim, g) = 1 - zmf(Y(:, dim), [a b]);
        end
    end

    % Reshape memberships into N x (m*G) feature matrix
    U = reshape(Memb, N, m*G);

    % Compute probabilistic recurrence: Gaussian on fuzzy features
    DistU = pdist2(U, U, 'euclidean');
    PRP = exp(-DistU.^2 / (2 * sigma_prp^2));
end

