function [Ur, sv] = approx_cca(stat, m, kappa, outdirname)

    if stat(length(stat)) ~= '/', stat = strcat(stat, '/'); end
    XYfile = strcat(stat, 'XY');
    Xfile = strcat(stat, 'X');
    Yfile = strcat(stat, 'Y');
    mapfile = strcat(stat, 'wordmap');
    
    disp('Loading data');
    tic;
    countXY = spconvert(load(XYfile));
    X = load(Xfile); 
    countX = spconvert([X(:,1) ones(size(X,1),1) X(:,2)]);
    vocab_size = size(countX, 1);
    Y = load(Yfile); 
    countY = spconvert([Y(:,1) ones(size(Y,1),1) Y(:,2)]);
    num_samples = sum(countX);
    mapfileID = fopen(mapfile);
    container = textscan(mapfileID, '%d %s');
    fclose(mapfileID);
    map = cell(vocab_size, 1);
    for i=1:vocab_size 
        map{container{1}(i)} = container{2}(i);
    end
    toc;
    
    disp('Computing the normalized covariance matrix Omega');
    tic;
    u = (countX + kappa) / num_samples;
    v = (countY + kappa) / num_samples;
    invdiagu = sparse(1:length(u), 1:length(u), u.^(-.5));
    invdiagv = sparse(1:length(v), 1:length(v), v.^(-.5));
    Omega = invdiagu * (countXY / num_samples) * invdiagv;
    
    fprintf('Omega: %d by %d (%d nonzeros)\n', ...
        size(Omega,1), size(Omega,2), nnz(Omega));
    [Ur, sv, ~] = svds(Omega, m);
    Ur = bsxfun(@times, Ur, 1./sqrt(sum(Ur.^2, 2)));
    toc;
    
    fprintf('Writing result to %s\n', outdirname);
    mkdir(outdirname);
    UrfileID = fopen(strcat(outdirname, '/Ur'), 'w');
    svfileID = fopen(strcat(outdirname, '/sv'), 'w');
    freqX = full(countX);
    [~, idx] = sort(freqX, 'descend');
    for itemp=1:vocab_size
        i = idx(itemp);
        fprintf(UrfileID, '%d %s', freqX(i), map{i}{1});
        row_length = norm(Ur(i,:));
        for j=1:m
            Ur(i,j) = Ur(i,j) / row_length;
            fprintf(UrfileID, ' %f', Ur(i,j));
        end
        fprintf(UrfileID, '\n');
    end
    for j=1:m
        fprintf(svfileID, '%f\n', sv(j,j));
    end
    fclose(UrfileID);
    fclose(svfileID);
end

