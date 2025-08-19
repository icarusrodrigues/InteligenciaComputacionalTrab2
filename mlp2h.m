function [STATS, TX_OK] = mlp2h(D, Nr, Ptrain, num_neuronios_oculta, taxa_aprendizado, epocas)
% mlp2h: Implementa e avalia um MLP com 2 camadas ocultas usando a funcao de ativacao ReLU.
%
%    Saidas:
%    STATS: Estatisticas de acerto [Media, Min, Max, Mediana, Std].
%    TX_OK: Vetor com as taxas de acerto de cada rodada.

    % Separação dos dados de entrada e saída
    Y = D(1, :);
    X = D(2:end, :);

    N = size(X, 2);
    Ntrn = floor(Ptrain * N / 100);
    Ntst = N - Ntrn;
    num_features = size(X, 1);

    TX_OK = zeros(1, Nr);

    for r = 1:Nr
        I = randperm(N);
        Xtrn_original = X(:, I(1:Ntrn));
        Ytrn_original = Y(:, I(1:Ntrn));
        Xtst = X(:, I(Ntrn+1:end))';
        Ytst_original = Y(:, I(Ntrn+1:end));

        % --- CONVERTA APENAS O SUBSET DE TREINAMENTO E OBTENHA O NUMERO DE CLASSES ---
        [Ytrn, num_classes] = convertToOneHot(Ytrn_original);
        Ytrn = Ytrn';

        Ytst = convertToOneHot(Ytst_original);
        Ytst = Ytst';

        Xtrn = Xtrn_original';

        % Inicialização dos pesos e biases (AGORA O NUMERO DE CLASSES É O CORRETO)
        W1 = randn(num_features, num_neuronios_oculta) * sqrt(2/num_features);
        b1 = zeros(1, num_neuronios_oculta);
        W2 = randn(num_neuronios_oculta, num_neuronios_oculta) * sqrt(2/num_neuronios_oculta);
        b2 = zeros(1, num_neuronios_oculta);
        W3 = randn(num_neuronios_oculta, num_classes) * 0.01;
        b3 = zeros(1, num_classes);

        for e = 1:epocas
            % Forward Propagation
            Z1 = Xtrn * W1 + b1; A1 = tanh(Z1);
            Z2 = A1 * W2 + b2; A2 = tanh(Z2);
            Z3 = A2 * W3 + b3; A3 = softmax(Z3);

            % Backpropagation
            delta3 = A3 - Ytrn;
            dW3 = A2' * delta3 / Ntrn; db3 = sum(delta3, 1) / Ntrn;
            delta2 = (delta3 * W3') .* (1 - A2.^2);
            dW2 = A1' * delta2 / Ntrn; db2 = sum(delta2, 1) / Ntrn;
            delta1 = (delta2 * W2') .* (1 - A1.^2);
            dW1 = Xtrn' * delta1 / Ntrn; db1 = sum(delta1, 1) / Ntrn;

            % Atualização dos pesos
            W1 = W1 - taxa_aprendizado * dW1; b1 = b1 - taxa_aprendizado * db1;
            W2 = W2 - taxa_aprendizado * dW2; b2 = b2 - taxa_aprendizado * db2;
            W3 = W3 - taxa_aprendizado * dW3; b3 = b3 - taxa_aprendizado * db3;
        end

        % Predição
        Z1_tst = Xtst * W1 + b1; A1_tst = max(0, Z1_tst);
        Z2_tst = A1_tst * W2 + b2; A2_tst = max(0, Z2_tst);
        Z3_tst = A2_tst * W3 + b3;
        Ypred = softmax(Z3_tst);

        % Avaliacao
        [~, pred_classes] = max(Ypred, [], 2);
        [~, real_classes] = max(Ytst, [], 2);
        acertos = sum(pred_classes == real_classes);
        TX_OK(r) = 100 * (acertos / Ntst);
    end

    STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
end

function P = softmax(Z)
    exp_Z = exp(Z - max(Z, [], 2));
    P = exp_Z ./ sum(exp_Z, 2);
end
