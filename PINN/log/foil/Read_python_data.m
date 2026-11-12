function [Xdata, Ydata, Udata, Vdata, Pdata, aoa, creal, cimag, radius] = Read_python_data(filename)
% 读取Python生成的MAT文件数据
%
% 输入:
%   filename - MAT文件名 (默认: 'matrix_data.mat')
%
% 输出:
%   matrix_data - 三维矩阵数据 [M, N, L]
%   a, b, c - 单独的元素值
%   M, N, L - 矩阵维度


    % 检查文件是否存在
    if ~exist(filename, 'file')
        error(['file ''', filename, ''' not exists.']);
    end
    
    % 加载数据
    data = load(filename);
    
    % 提取数据
    Xdata = data.XX;
    Ydata = data.YY;
    Udata = data.UU;
    Vdata = data.VV;
    Pdata = data.PP;
    aoa = data.aoa;
    creal = data.creal;
    cimag = data.cimag;
    radius = data.R;
   
end


