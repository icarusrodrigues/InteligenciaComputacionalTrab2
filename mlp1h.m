function [STATS, TX_OK] = mlp1h(D, Nr, Ptrain, num_neuronios_oculta, taxa_aprendizado, epocas)
% mlp1h: Implementa e avalia um MLP com 1 camada oculta.
%
%   Saidas:
%   STATS: Estatisticas de acerto [Media, Min, Max, Mediana, Std].
%   TX_OK: Vetor com as taxas de acerto de cada rodada.

    % Separação dos dados de entrada e saída
    Y = D(1, :);
    X = D(2:end, :);

    N = size(X, 2);
    Ntrn = floor(Ptrain * N / 100);
    Ntst = N - Ntrn;
    num_features = size(X, 1);

    TX_OK = zeros(1, Nr);

    for r = 1:Nr
        % Embaralhamento e separação
        I = randperm(N);
        Xtrn_original = X(:, I(1:Ntrn));
        Ytrn_original = Y(:, I(1:Ntrn));
        Xtst = X(:, I(Ntrn+1:end))';
        Ytst_original = Y(:, I(Ntrn+1:end));

        [Ytrn, num_classes] = convertToOneHot(Ytrn_original);
        Ytrn = Ytrn'; % Transpor para que cada linha seja um exemplo

        Ytst = convertToOneHot(Ytst_original);
        Ytst = Ytst'; % Transpor para que cada linha seja um exemplo

        Xtrn = Xtrn_original';

        % Inicialização dos pesos e biases (AGORA O NUMERO DE CLASSES É O CORRETO)
        W1 = randn(num_features, num_neuronios_oculta) * sqrt(2/num_features);
        b1 = zeros(1, num_neuronios_oculta);
        W2 = randn(num_neuronios_oculta, num_classes) * 0.01;
        b2 = zeros(1, num_classes);

        % Loop de treinamento (Backpropagation)
        for e = 1:epocas
            % Forward Propagation
            Z1 = Xtrn * W1 + b1;
            A1 = max(0, Z1); % <--- FUNÇÃO DE ATIVAÇÃO ReLU
            Z2 = A1 * W2 + b2;
            A2 = softmax(Z2);

            % Backpropagation
            delta2 = A2 - Ytrn;
            dW2 = A1' * delta2 / Ntrn;
            db2 = sum(delta2, 1) / Ntrn;

            % --- DERIVADA DA ReLU ---
            % Onde Z1 > 0, o gradiente eh 1; caso contrario, eh 0.
            delta1 = (delta2 * W2') .* (Z1 > 0);
            % ------------------------

            dW1 = Xtrn' * delta1 / Ntrn;
            db1 = sum(delta1, 1) / Ntrn;

            % Atualização dos pesos
            W1 = W1 - taxa_aprendizado * dW1;
            b1 = b1 - taxa_aprendizado * db1;
            W2 = W2 - taxa_aprendizado * dW2;
            b2 = b2 - taxa_aprendizado * db2;
        end

        % Predição
        Z1_tst = Xtst * W1 + b1;
        A1_tst = tanh(Z1_tst);
        Z2_tst = A1_tst * W2 + b2;
        Ypred = softmax(Z2_tst);

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
