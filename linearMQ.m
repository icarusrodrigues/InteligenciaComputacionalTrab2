function [STATS, TX_OK, W] = linearMQ(D, Nr, Ptrain)

    % Separação dos dados de entrada e saída
    Y = D(1, :);
    X = D(2:end, :);

    [Y_one_hot, num_classes] = convertToOneHot(Y);

    N = size(X, 2);
    Ntrn = floor(Ptrain * N / 100);
    Ntst = N - Ntrn;

    TX_OK = zeros(1, Nr);

    % Define um valor de regularização FIXO
    lambda = 0.01;

    for r = 1:Nr
        % Embaralhamento dos dados
        I = randperm(N);
        X_shuffled = X(:, I);
        Y_shuffled = Y_one_hot(:, I);

        % Separação em dados de treino e teste
        Xtrn = X_shuffled(:, 1:Ntrn);
        Ytrn = Y_shuffled(:, 1:Ntrn);
        Xtst = X_shuffled(:, Ntrn+1:end);
        Ytst = Y_shuffled(:, Ntrn+1:end);

        % --- NORMALIZAÇÃO Z-SCORE ---
        media_X = mean(Xtrn, 2);
        std_X = std(Xtrn, 0, 2);
        std_X(std_X == 0) = 1e-6;
        Xtrn_norm = (Xtrn - media_X) ./ std_X;
        Xtst_norm = (Xtst - media_X) ./ std_X;
        % ------------------------------

        % --- REGULARIZAÇÃO DE TIKHONOV (AGORA COM LAMBDA FIXO) ---
        W = Ytrn * Xtrn_norm' * inv(Xtrn_norm * Xtrn_norm' + lambda * eye(size(Xtrn_norm, 1)));
        % ---------------------------------

        % Predição
        Ypred = W * Xtst_norm;

        % Avaliação
        Nacertos = evalclassifier(Ytst, Ypred, Ntst);
        TX_OK(r) = 100 * (Nacertos / Ntst);
    end

    STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
    W = []; % A matriz W não é mais um retorno principal
end
