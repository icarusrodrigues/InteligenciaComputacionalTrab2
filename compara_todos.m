clear; clc;

pkg load statistics;

Nr=50; % No. de repeticoes
Ptrain=80; % Porcentagem de treinamento

% --- PARAMETROS ADICIONAIS PARA OS NOVOS CLASSIFICADORES ---
% MLP 1 Camada
num_neuronios_oculta_MLP1 = 50; % Pode ser ajustado
taxa_aprendizado_MLP1 = 0.001;
epocas_MLP1 = 500;

% MLP 2 Camadas
num_neuronios_oculta_MLP2 = 30; % Pode ser ajustado
taxa_aprendizado_MLP2 = 0.0001;
epocas_MLP2 = 2000;

% --- EXECUCAO DOS CLASSIFICADORES ---

% 1. Classificador Linear de Minimos Quadrados (MQ)
D=load('recfaces.dat'); % Recarrega os dados
tic; [STATS_1 TX_OK1 W_MQ]=linearMQ(D,Nr,Ptrain); Tempo1=toc;

% 2. Perceptron Logistico (PL)
D=load('recfaces.dat'); % Recarrega os dados
tic; [STATS_2 TX_OK2 W_PL]=perceptronLogistico(D,Nr,Ptrain,taxa_aprendizado_MLP1, 100); Tempo2=toc;

% 3. Perceptron Multicamadas (MLP-1H)
D=load('recfaces.dat'); % Recarrega os dados
% Exemplo de como normalizar os dados X
X = D(2:end, :); % Carregue X

media_X = mean(X, 2);
std_X = std(X, 0, 2);

% Lembre-se de nao dividir por zero
std_X(std_X == 0) = 1e-6;

X_normalizado = (X - media_X) ./ std_X;
D(2:end, :) = X_normalizado; % Substitua os dados originais pelos normalizados
tic; [STATS_3 TX_OK3]=mlp1h(D,Nr,Ptrain,num_neuronios_oculta_MLP1,taxa_aprendizado_MLP1, epocas_MLP1); Tempo3=toc;

% 4. Perceptron Multicamadas (MLP-2H)
D=load('recfaces.dat'); % Recarrega os dados
% Exemplo de como normalizar os dados X
X = D(2:end, :); % Carregue X

media_X = mean(X, 2);
std_X = std(X, 0, 2);

% Lembre-se de nao dividir por zero
std_X(std_X == 0) = 1e-6;

X_normalizado = (X - media_X) ./ std_X;
D(2:end, :) = X_normalizado; % Substitua os dados originais pelos normalizados
tic; [STATS_4 TX_OK4]=mlp2h(D,Nr,Ptrain,num_neuronios_oculta_MLP2,taxa_aprendizado_MLP2, epocas_MLP2); Tempo4=toc;

% --- EXIBICAO DOS RESULTADOS ---

fprintf('Resultados do Classificador Linear de Minimos Quadrados (MQ):\n');
disp(STATS_1);
fprintf('Tempo de execução: ');
disp(Tempo1);

fprintf('\nResultados do Perceptron Logistico (PL):\n');
disp(STATS_2);
fprintf('Tempo de execução: ');
disp(Tempo2);

fprintf('\nResultados do MLP com 1 Camada Oculta (MLP-1H):\n');
disp(STATS_3);
fprintf('Tempo de execução: ');
disp(Tempo3);

fprintf('\nResultados do MLP com 2 Camadas Ocultas (MLP-2H):\n');
disp(STATS_4);
fprintf('Tempo de execução: ');
disp(Tempo4);

TEMPOS=[Tempo1 Tempo2 Tempo3 Tempo4];

% --- VISUALIZACAO DOS RESULTADOS ---

%boxplot([TX_OK1' TX_OK2' TX_OK3' TX_OK4'])
%set(gca (), "xtick", [1 2 3 4], "xticklabel", {"MQ","PL", "MLP-1H","MLP-2H"})
%title('Comparacao de Classificadores');
%xlabel('Classificador');
%ylabel('Taxas de acerto');
