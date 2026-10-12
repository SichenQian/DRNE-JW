function write_full_precision(filename, record)
%WRITE_FULL_PRECISION Export small records using 17 significant digits.
% Large arrays are summarized explicitly; complete arrays remain in MAT files.
fid=fopen(filename,'w');assert(fid>=0,'Cannot create precision record.');
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'Full-precision numerical record (17 significant digits)\n');
fprintf(fid,'MATLAB %s; platform %s\n\n',version,computer);
visit(record,'record');
    function visit(v,path)
        if istable(v),visit(table2struct(v),path);return;end
        if isstruct(v)
            f=fieldnames(v);
            for j=1:numel(v)
                for k=1:numel(f),visit(v(j).(f{k}),sprintf('%s(%d).%s',path,j,f{k}));end
            end
        elseif iscell(v)
            for j=1:numel(v),visit(v{j},sprintf('%s{%d}',path,j));end
        elseif isnumeric(v)||islogical(v)
            if numel(v)<=5000 && ismatrix(v)
                fprintf(fid,'%s = %s\n',path,mat2str(v,17));
            else
                fprintf(fid,'%s: array size %s; full values retained in MAT output\n',path,mat2str(size(v)));
                if ~isempty(v),fprintf(fid,'  min=%.17g; max=%.17g\n',min(v(:)),max(v(:)));end
            end
        elseif ischar(v)||isstring(v)
            fprintf(fid,'%s = %s\n',path,strjoin(string(v(:)),' | '));
        end
    end
end
