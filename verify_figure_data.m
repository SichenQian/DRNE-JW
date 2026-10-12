function report=verify_figure_data(outDir)
%VERIFY_FIGURE_DATA Compare freshly generated graphics data with results/.
% Run verify_reproduction first. Formatting and export metadata are ignored.
repo=fileparts(mfilename('fullpath'));
if nargin<1||isempty(outDir),outDir=fullfile(repo,'reproduced_results');end
referenceDir=fullfile(repo,'results');
files=dir(fullfile(referenceDir,'**','*.fig'));reports={};
for k=1:numel(files)
    reference=fullfile(files(k).folder,files(k).name);
    relative=reference(numel(referenceDir)+2:end);
    if startsWith(relative,['verification' filesep]),continue;end
    generated=fullfile(outDir,relative);
    assert(isfile(generated),'Missing generated figure: %s',relative);
    item=compare_figure_records(reference,generated);
    item.referenceFile=strrep(fullfile('results',relative),filesep,'/');
    item.generatedFile=strrep(relative,filesep,'/');
    reports{end+1}=item; %#ok<AGROW>
end
assert(~isempty(reports),'No archived reference figures were found.');
report=struct('figureCount',numel(reports),'figures',{reports}, ...
    'allReferenceObjectsExact',all(cellfun(@(x)x.allReferenceObjectsExact,reports)), ...
    'allGraphicsDataExact',all(cellfun(@(x)x.allObjectsExact,reports)));
fid=fopen(fullfile(outDir,'figure_reference_comparison.json'),'w');assert(fid>=0);
c=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(report,'PrettyPrint',true));
assert(report.allGraphicsDataExact,'Fresh figure data differ from the archived reference.');
fprintf('FIGURE_DATA_PASS: %d figures compared exactly\n',numel(reports));
end

function report=compare_figure_records(referenceFile,generatedFile)
%COMPARE_FIGURE_RECORDS Match numerical graphics data, ignoring formatting.
a=openfig(referenceFile,'invisible');b=openfig(generatedFile,'invisible');
clean=onCleanup(@()close([a,b])); %#ok<NASGU>
aa=objects(a);bb=objects(b);used=false(numel(bb),1);
entries=struct('type',{},'points',{},'exact',{},'maxAbsoluteDifference',{});
for j=1:numel(aa)
    best=Inf; selected=0;
    for k=1:numel(bb)
        if used(k)||~strcmp(aa{j}.type,bb{k}.type),continue;end
        fields=fieldnames(aa{j}.data); d=0;
        for f=1:numel(fields)
            key=fields{f};u=aa{j}.data.(key);
            if ~isfield(bb{k}.data,key)||~isequal(size(u),size(bb{k}.data.(key))),d=Inf;break;end
            v=bb{k}.data.(key);
            if ~isequal(isnan(u),isnan(v))||~isequal(isinf(u),isinf(v)),d=Inf;break;end
            finite=isfinite(u)&isfinite(v);
            if ~isequaln(u(~finite),v(~finite)),d=Inf;break;end
            delta=abs(u(finite)-v(finite));d=max(d,max([0;delta(:)]));
        end
        if d<best,best=d;selected=k;end
    end
    if selected>0,used(selected)=true;end
    entries(end+1)=struct('type',aa{j}.type,'points',numel(aa{j}.data.XData), ...
        'exact',best==0,'maxAbsoluteDifference',best); %#ok<AGROW>
end
report=struct('referenceFile',referenceFile,'generatedFile',generatedFile, ...
    'referenceObjects',numel(aa),'generatedObjects',numel(bb), ...
    'exactlyMatchedObjects',sum([entries.exact]),'allReferenceObjectsExact',all([entries.exact]), ...
    'allObjectsExact',all([entries.exact])&&numel(aa)==numel(bb), ...
    'maxAbsoluteDifference',max([0,entries.maxAbsoluteDifference]),'objects',entries);
end

function records=objects(fig)
h=findall(fig);records={};
types={'line','patch','scatter','errorbar','bar','surface'};
props={'XData','YData','ZData','YPositiveDelta','YNegativeDelta','XPositiveDelta','XNegativeDelta'};
for j=1:numel(h)
    if ~ismember(h(j).Type,types)||~isprop(h(j),'XData')||isempty(h(j).XData),continue;end
    if ~any(isfinite(h(j).XData(:))),continue;end
    data=struct();
    for k=1:numel(props)
        if isprop(h(j),props{k}),data.(props{k})=double(h(j).(props{k}));end
    end
    records{end+1}=struct('type',h(j).Type,'data',data); %#ok<AGROW>
end
end
