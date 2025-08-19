function [STATS, TX_OK, W] = perceptronLogistico(D, Nr, Ptrain, taxa_aprendizado, epocas)
% perceptronLogistico: Implementa e avalia um classificador Perceptron Logistico.
%
%   Entradas:
%   D, Nr, Ptrain: Dados e parametros de simulacao.
%   taxa_aprendizado: Taxa de aprendizado para o Gradiente Descendente.
%   epocas: Numero de epocas de treinamento.
%
%   Saidas:
%   STATS, TX_OK, W: Estatisticas de desempenho.

    % Separação dos dados de entrada e saída
    Y = D(1, :)'; % Rótulos como vetor coluna
    X = D(2:end, :)'; % Features como matriz onde cada linha é um exemplo

    % Normalizacao dos dados (opcional, mas recomendado)
    % X = (X - mean(X)) ./ std(X);

    N = size(X, 1);
    Ntrn = floor(Ptrain * N / 100);
    Ntst = N - Ntrn;

    TX_OK = zeros(1, Nr);

    for r = 1:Nr
        % Embaralhamento
        I = randperm(N);
        X_shuffled = X(I, :);
        Y_shuffled = Y(I, :);

        Xtrn = X_shuffled(1:Ntrn, :);
        Ytrn = Y_shuffled(1:Ntrn, :);
        Xtst = X_shuffled(Ntrn+1:end, :);
        Ytst = Y_shuffled(Ntrn+1:end, :);

        % Treinamento do Perceptron Logistico (binario)
        % Adiciona um bias
        Xtrn_bias = [ones(Ntrn, 1) Xtrn];
        W = zeros(size(Xtrn_bias, 2), 1);

        % Loop de treinamento
        for e = 1:epocas
            Z = Xtrn_bias * W;
            H = 1 ./ (1 + exp(-Z)); % Funcao Sigmoide
            gradiente = Xtrn_bias' * (H - Ytrn) / Ntrn;
            W = W - taxa_aprendizado * gradiente;
        end

        % Predição
        Xtst_bias = [ones(Ntst, 1) Xtst];
        Z_pred = Xtst_bias * W;
        Ypred = (Z_pred > 0.5); % Limiar 0.5 para classificacao

        % Avaliacao (assumindo Ytst e Ypred sao vetores coluna de 0 e 1)
        acertos = sum(Ypred == Ytst);
        TX_OK(r) = 100 * (acertos / Ntst);
    end

    STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
end
