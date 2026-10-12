function report = compare_numeric_records(reference, actual, absoluteTolerance, relativeTolerance)
%COMPARE_NUMERIC_RECORDS Compare every numeric/logical leaf of a reference.
% Runtime and worker-count metadata are excluded explicitly. Added fields in
% the new record are allowed; missing reference fields are reported as errors.
if nargin<3, absoluteTolerance=1e-11; end
if nargin<4, relativeTolerance=1e-9; end
entries=struct('path',{},'count',{},'maxAbsoluteDifference',{}, ...
    'maxRelativeDifference',{},'identical',{},'pass',{});
excluded=strings(0,1); missing=strings(0,1);
visit(reference,actual,'root');
report=struct('numericLeaves',numel(entries),'numericValues',sum([entries.count]), ...
    'absoluteTolerance',absoluteTolerance,'relativeTolerance',relativeTolerance, ...
    'maxAbsoluteDifference',max([0,entries.maxAbsoluteDifference]), ...
    'maxRelativeDifference',max([0,entries.maxRelativeDifference]), ...
    'numericallyIdentical',all([entries.identical])&&isempty(missing), ...
    'pass',all([entries.pass])&&isempty(missing),'excludedMetadata',{cellstr(excluded)}, ...
    'missingFields',{cellstr(missing)},'leaves',entries);
    function visit(a,b,path)
        if istable(a)
            if ~istable(b), missing(end+1)=string(path); return; end
            visit(table2struct(a),table2struct(b),path); return;
        end
        if isstruct(a)
            if ~isstruct(b)||~isequal(size(a),size(b)), missing(end+1)=string(path);return;end
            fn=fieldnames(a);
            for j=1:numel(a)
                for k=1:numel(fn)
                    key=fn{k}; pp=sprintf('%s(%d).%s',path,j,key);
                    if ~isempty(regexpi(key,'runtime|wallseconds|parallelworkers|useparallel','once'))
                        excluded(end+1)=string(pp); continue;
                    end
                    if ~isfield(b,key),missing(end+1)=string(pp);continue;end
                    visit(a(j).(key),b(j).(key),pp);
                end
            end
        elseif iscell(a)
            if ~iscell(b)||~isequal(size(a),size(b)),missing(end+1)=string(path);return;end
            for j=1:numel(a),visit(a{j},b{j},sprintf('%s{%d}',path,j));end
        elseif isnumeric(a)||islogical(a)
            sameShape=(isnumeric(b)||islogical(b))&&isequal(size(a),size(b));
            exact=sameShape&&isequaln(a,b); ad=Inf; rd=Inf; pass=false;
            if sameShape
                aa=double(a(:));bb=double(b(:)); fin=isfinite(aa)&isfinite(bb);
                sameNonfinite=isequaln(aa(~fin),bb(~fin));
                delta=abs(aa(fin)-bb(fin)); scale=max(abs(aa(fin)),abs(bb(fin)));
                ad=max([0;delta]);rd=max([0;delta./max(1,scale)]);
                pass=sameNonfinite&&all(delta<=absoluteTolerance+relativeTolerance*scale);
                if islogical(a),pass=exact;end
            end
            entries(end+1)=struct('path',path,'count',numel(a), ...
                'maxAbsoluteDifference',ad,'maxRelativeDifference',rd,'identical',exact,'pass',pass);
        end
    end
end
