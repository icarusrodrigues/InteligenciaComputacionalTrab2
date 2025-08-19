function [Y_one_hot, num_classes] = converterParaOneHot(Y)
 % Converte um vetor de rotulos de classe (sejam eles inteiros ou floats)
% para a representacao one-hot.
%
%   Y: Vetor de rotulos de classe (ex: [1 2 1 3] ou [0.1 0.2 0.1 0.3]).
%
%   Y_one_hot: Matriz com a representacao one-hot.
%   num_classes: Numero total de classes identificadas.

    % Encontra os rotulos de classe unicos e os mapeia para inteiros
    [classes, ~, Y_inteiros] = unique(Y);

    num_classes = length(classes);
    num_exemplos = size(Y, 2);

    Y_one_hot = zeros(num_classes, num_exemplos);

    for i = 1:num_exemplos
        Y_one_hot(Y_inteiros(i), i) = 1;
    end
end
