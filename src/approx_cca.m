function [ Ur sv ] = approx_cca(stats, m, kappa)
    temp = regexp(stats,'(.*)window(\d)(.*)','tokens','once');
    num_duplicates = str2num(temp{2})-1;
    countXY = spconvert(load(stats));
    countX = sum(countXY, 2);
    countY = sum(countXY, 1)';
    
    
    Ur = 0;
    sv = 0;
end

