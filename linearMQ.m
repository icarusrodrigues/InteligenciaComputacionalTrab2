function [STATS, TX_OK, W] = linearMQ(D, Nr, Ptrain)
    % Separação dos dados de entrada e saída
    Y = D(1, :); % Primeira linha eh o Y (rotulos)
    X = D(2:end, :); % O restante sao as features (X)

    [Y_one_hot, num_classes] = convertToOneHot(Y);

    N = size(X, 2); % Numero de exemplos
    Ntrn = floor(Ptrain * N / 100); % Numero de exemplos de treinamento
    Ntst = N - Ntrn; % Numero de exemplos de teste

    TX_OK = zeros(1, Nr);

    for r = 1:Nr
        % Embaralhamento dos dados
        I = randperm(N);
        X = X(:, I);
        Y = Y_one_hot(:, I);

        % Separação em dados de treino e teste
        Xtrn = X(:, 1:Ntrn);
        Ytrn = Y_one_hot(:, 1:Ntrn);
        Xtst = X(:, Ntrn+1:end);
        Ytst = Y_one_hot(:, Ntrn+1:end);

        % Treinamento do classificador de Minimos Quadrados
        W = Ytrn / Xtrn;

        % Predição
        Ypred = W * Xtst;

        % Avaliacao
        Nacertos = evalclassifier(Ytst, Ypred, Ntst);
        TX_OK(r) = 100 * (Nacertos / Ntst);
    end

    % Estatisticas
    STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
end
