function [Ur, sv] = approx_cca(stat, m, kappa, outdirname)

    XYfile = fullfile(stat, 'XY');
    Xfile  = fullfile(stat, 'X');
    Yfile  = fullfile(stat, 'Y');
    
    mkdir(outdirname);
    logf = fopen(fullfile(outdirname, 'log'), 'w');
    fprintf(logf, '[tic]\nLoading data from %s\n', stat);
    tic;
    countXY = spconvert(load(XYfile));
    X = load(Xfile); 
    countX = spconvert([X(:,1) ones(size(X,1),1) X(:,2)]);
    Y = load(Yfile); 
    countY = spconvert([Y(:,1) ones(size(Y,1),1) Y(:,2)]);
    num_samples = sum(countX);
    loadtime = toc;
    fprintf(logf, '[toc] %f seconds\n\n', loadtime);
    
    fprintf(logf, '[tic]\nComputing Omega');
    tic;
    u = (countX + kappa) / num_samples;
    v = (countY + kappa) / num_samples;
    invdiagu = sparse(1:length(u), 1:length(u), u.^(-.5));
    invdiagv = sparse(1:length(v), 1:length(v), v.^(-.5));
    Omega = invdiagu * (countXY / num_samples) * invdiagv;
    fprintf(logf, ' %d x %d (%d nonzeros)\n', ...
        size(Omega,1), size(Omega,2), nnz(Omega));
    
    fprintf(logf, 'Computing top %d SVD components\n', m);
    opts.issym = 1;
    [Ur, sv, flag] = eigs(@Afun, size(Omega,1), m, 'lm', opts);
    if flag == 0, conv = 'yes'; else conv = 'no'; end
    fprintf(logf, 'All eigenvalues converged? %s\n', conv);
    sv = sqrt(diag(sv)); 
    fprintf(logf, 'Normalizing rows of the left singular vector matrix\n');
    Ur = bsxfun( @rdivide, Ur, sqrt(sum(Ur.^2, 2)) ); 
    Ur(~isfinite(Ur)) = 1;
    runtime = toc;
    fprintf(logf, '[toc] %f seconds\n\n', runtime);
    
    fprintf(logf, '[tic]\nWriting result to %s\n', outdirname);
    tic;
    save(fullfile(outdirname, 'Ur'), 'Ur', '-ascii', '-double');
    save(fullfile(outdirname, 'sv'), 'sv', '-ascii', '-double');
    writetime = toc;
    fprintf(logf, '[toc] %f seconds\n', writetime);
    
    fclose(logf);
    exit;
    function y = Afun(x)
        y = Omega * (Omega' * x);
    end
end

