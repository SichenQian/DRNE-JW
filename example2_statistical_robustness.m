%% Example 2: eight-player edge-service game
% Finite-sample coverage, held-out equilibrium gaps, and budget sensitivity.

clear; clc; close all;
rng(260827, 'twister');

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
outDir = fullfile(scriptDir, 'results', 'example2');
if ~isfolder(outDir)
    mkdir(outDir);
end

%% Game parameters
p.N = 8;
p.a = [4.15;4.55;4.90;4.30;5.05;4.65;4.35;5.15];
p.r = [2.35;2.70;2.90;2.45;2.95;2.62;2.50;3.00];
p.c = [0.21;0.28;0.25;0.19;0.30;0.24;0.22;0.27];
p.wL = [0.72;0.68;0.75;0.70;0.66;0.73;0.69;0.71];
p.wU = [1.29;1.34;1.25;1.31;1.36;1.28;1.33;1.30];
p.B = zeros(p.N);
for i = 1:p.N
    j1 = mod(i, p.N) + 1;
    j2 = mod(i+2, p.N) + 1;
    p.B(i,j1) = 0.28 + 0.025*mod(i,3);
    p.B(i,j2) = 0.19 + 0.020*mod(i+1,4);
end
p.B(2,7) = 0.17; p.B(6,1) = 0.16;

% Communication edges and physical interaction weights are distinct.
p.Acomm = zeros(p.N);
for i = 1:p.N
    j = mod(i,p.N)+1;
    p.Acomm(i,j)=1; p.Acomm(j,i)=1;
end
p.L = diag(sum(p.Acomm,2))-p.Acomm;
lapEig = sort(real(eig(p.L)));
p.lambda2 = lapEig(2); p.lambdaN = lapEig(end);

p.lambdaXi = [12.0;12.5;11.8;12.2;12.7;12.1;11.9;12.4];
p.lambdaY  = [12.0;12.5;11.8;12.2;12.7;12.1;11.9;12.4];
p.sigmaY = 0.14;
p.rhoCover = p.sigmaY*sqrt((p.N-1)/3);
p.rhoBase = 1.05*p.rhoCover*ones(p.N,1);
p.kappaReference = 0.60;

Kgrid = [5 10 20 40 80 160 320 640 1280];
p.deltaTotal = 0.05;
p.deltaLocal = p.deltaTotal/p.N;

% Local random streams give identical samples in serial and parallel runs.
useParallel = exist('parpool','file')==2 && ...
    license('test','Distrib_Computing_Toolbox');
parallelWorkers = 0;
if useParallel
    pool = gcp('nocreate');
    if isempty(pool)
        parallelWorkers = min(10,feature('numcores'));
        pool = parpool('local',parallelWorkers);
    else
        parallelWorkers = pool.NumWorkers;
    end
    fprintf('Parallel execution enabled with %d workers.\n',parallelWorkers);
else
    fprintf(['Parallel Computing Toolbox unavailable; using deterministic ', ...
        'serial execution with vectorized W2 batches.\n']);
end

%% Uniform theorem conditions
% Uniform constants depend on supports, multipliers, and sample size.
[cst0, ~] = analytical_constants_base(p);
p.muZi = 2*min(p.lambdaXi,p.lambdaY)-cst0.ellZ-0.50;
[cst0, cert] = analytical_constants_base(p);
theoryByK = cell(numel(Kgrid),1);
gainByK = cell(numel(Kgrid),1);
auditByK = cell(numel(Kgrid),1);
for kk = 1:numel(Kgrid)
    pK = p; pK.K = Kgrid(kk)*ones(p.N,1);
    [cstK, certK] = analytical_constants(pK, cert);
    [gainK,cstK] = select_theorem_gains(pK,cstK,certK);
    auditK = theory_audit(pK,cstK,certK,gainK,p.rhoBase);
    assert_all_pass(auditK);
    theoryByK{kk}=cstK; gainByK{kk}=gainK; auditByK{kk}=auditK;
end

%% Wasserstein-radius calibration and coverage validation
nCalibration = 5000;
nCoverage = 3000;
supportWidth=p.wU-p.wL;
[widest,widestPlayer]=max(supportWidth);
qNorm = zeros(p.N,numel(Kgrid));
medianNorm = zeros(numel(Kgrid),1);
iqrNorm = zeros(numel(Kgrid),2);
epsilonByK = zeros(p.N,numel(Kgrid));
coverageProb = zeros(numel(Kgrid),1);
coverageCI = zeros(numel(Kgrid),2);
quantileLevels = [0.10 0.25 0.50 0.75 0.90];
w2QuantileNorm = zeros(numel(Kgrid),numel(quantileLevels));
w2QuantileNormByPlayer = zeros(p.N,numel(Kgrid),numel(quantileLevels));
w2CalibrationNorm = cell(numel(Kgrid),1);
coverageW2Norm = cell(numel(Kgrid),1);
coverageIndicators = cell(numel(Kgrid),1);
calibrationSeeds = zeros(numel(Kgrid),p.N);
coverageSeeds = zeros(numel(Kgrid),p.N);
for kk=1:numel(Kgrid)
    calibrationSeeds(kk,:)=710000+1000*kk+(1:p.N);
    coverageSeeds(kk,:)=720000+1000*kk+(1:p.N);
end

fprintf('\nCalibrating empirical Uniform-W2 radii and validating coverage...\n');
ticStatistical = tic;
for kk = 1:numel(Kgrid)
    K = Kgrid(kk);
    % Calibrate playerwise quantiles with independent random streams.
    calibrationW2 = zeros(nCalibration,p.N);
    validationW2 = zeros(nCoverage,p.N);
    for i = 1:p.N
        calibrationW2(:,i)=uniform_w2_replicates( ...
            K,nCalibration,calibrationSeeds(kk,i));
        qNorm(i,kk)=quantile(calibrationW2(:,i),1-p.deltaLocal);
        epsilonByK(i,kk)=supportWidth(i)*qNorm(i,kk);
        w2QuantileNormByPlayer(i,kk,:)=reshape(quantile( ...
            calibrationW2(:,i),quantileLevels),1,1,[]);
        validationW2(:,i)=uniform_w2_replicates( ...
            K,nCoverage,coverageSeeds(kk,i));
    end
    w2CalibrationNorm{kk}=calibrationW2;
    w2QuantileNorm(kk,:)=reshape( ...
        w2QuantileNormByPlayer(widestPlayer,kk,:),1,[]);
    medianNorm(kk)=w2QuantileNorm(kk,3);
    iqrNorm(kk,:)=w2QuantileNorm(kk,[2 4]);
    covered = all(validationW2<=qNorm(:,kk).'+10*eps(qNorm(:,kk).'),2);
    coverageW2Norm{kk}=validationW2;
    coverageIndicators{kk}=covered;
    successes = sum(covered);
    coverageProb(kk)=successes/nCoverage;
    coverageCI(kk,:)=wilson_interval(successes,nCoverage,1.96);
    fprintf('  K=%4d: radius=%.5f, joint coverage=%.4f\n', ...
        K,max(epsilonByK(:,kk)),coverageProb(kk));
end
w2QuantileMaxSupport=widest*w2QuantileNorm;
w2QuantileByPlayer=zeros(p.N,numel(Kgrid),numel(quantileLevels));
for qq=1:numel(quantileLevels)
    w2QuantileByPlayer(:,:,qq)=supportWidth.* ...
        w2QuantileNormByPlayer(:,:,qq);
end
representativePlayer=widestPlayer;
representativeW2Quantiles=squeeze( ...
    w2QuantileByPlayer(representativePlayer,:,:));
representativeRadius=epsilonByK(representativePlayer,:);
representativeMedianW2=representativeW2Quantiles(:,3).';
representativeW2IQR=representativeW2Quantiles(:,[2 4]);
maxPlayerRadius=max(epsilonByK,[],1);
maxPlayerW2Quantiles=squeeze(max(w2QuantileByPlayer,[],1));
maxPlayerMedianW2=maxPlayerW2Quantiles(:,3).';
maxPlayerW2IQR=maxPlayerW2Quantiles(:,[2 4]);

%% Baseline dataset and monotonicity check
baseIndex = find(Kgrid==40,1);
Kbase = Kgrid(baseIndex);
baseDatasetSeed = 44100;
baseSamples = generate_datasets(p,Kbase,baseDatasetSeed);
pBase = p;
pBase.K = Kbase*ones(p.N,1);
pBase.samples = baseSamples;
pBase.epsilon = epsilonByK(:,baseIndex);
pBase.rho = p.rhoBase;
[numericMuObserved,numericAuditPoints] = numerical_monotonicity_audit(pBase,500);
if numericMuObserved <= 0
    error('DRGame:NumericalMonotonicityAudit', ...
        'The supplemental Jacobian audit found a nonpositive eigenvalue.');
end
[xProxyBase,proxyBaseInfo] = solve_proxy_ne(pBase,nominal_ne(pBase),2e-10,8000);

fprintf('\n--- INITIAL THEORY AUDIT (K=40) ---\n');
fprintf('certified mu_F %.6e; observed Jacobian minimum %.6e (%d points)\n', ...
    cert.muF,numericMuObserved,numericAuditPoints);
fprintf('lambda_2(L)=%.6e, strategic rho/rho_cover=%.4f\n', ...
    p.lambda2,p.rhoBase(1)/p.rhoCover);
fprintf('All K-specific theorem audits: PASS (K=%s).\n',mat2str(Kgrid));

%% Sample-size study and held-out SNE gaps
nEquilibriumRep = 100;
nHeldout = 20000;
sampleGap = nan(nEquilibriumRep,numel(Kgrid));
sampleService = nan(nEquilibriumRep,numel(Kgrid));
sampleRisk = nan(nEquilibriumRep,numel(Kgrid));
sampleResidual = nan(nEquilibriumRep,numel(Kgrid));
sampleX = nan(p.N,nEquilibriumRep,numel(Kgrid));
equilibriumDatasetSeeds=zeros(nEquilibriumRep,numel(Kgrid));
heldoutSeeds=zeros(nEquilibriumRep,numel(Kgrid));
for kk=1:numel(Kgrid)
    equilibriumDatasetSeeds(:,kk)=810000+10000*kk+(1:nEquilibriumRep).';
    heldoutSeeds(:,kk)=910000+10000*kk+(1:nEquilibriumRep).';
end

fprintf('Running %d proxy-equilibrium repetitions for each K...\n',nEquilibriumRep);
for kk = 1:numel(Kgrid)
    K=Kgrid(kk);
    epsilonCol=epsilonByK(:,kk);
    datasetSeedCol=equilibriumDatasetSeeds(:,kk);
    heldoutSeedCol=heldoutSeeds(:,kk);
    gapCol=nan(nEquilibriumRep,1); serviceCol=gapCol;
    riskCol=gapCol; residualCol=gapCol;
    xCol=nan(p.N,nEquilibriumRep);
    if useParallel
        parfor rep = 1:nEquilibriumRep
            [xCol(:,rep),gapCol(rep),serviceCol(rep),riskCol(rep), ...
                residualCol(rep)]=run_sample_replication(p,K, ...
                epsilonCol,nHeldout,datasetSeedCol(rep),heldoutSeedCol(rep));
        end
    else
        for rep = 1:nEquilibriumRep
            [xCol(:,rep),gapCol(rep),serviceCol(rep),riskCol(rep), ...
                residualCol(rep)]=run_sample_replication(p,K, ...
                epsilonCol,nHeldout,datasetSeedCol(rep),heldoutSeedCol(rep));
        end
    end
    sampleX(:,:,kk)=xCol;
    sampleGap(:,kk)=gapCol;
    sampleService(:,kk)=serviceCol;
    sampleRisk(:,kk)=riskCol;
    sampleResidual(:,kk)=residualCol;
    fprintf('  K=%4d complete: median held-out gap %.6e\n',K,median(gapCol));
end
if max(sampleResidual,[],'all')>2e-7
    error('DRGame:SampleProxyResidual','A sample-size proxy solve has excessive residual.');
end
sampleGapMedian=median(sampleGap,1);
sampleGapIQR=quantile(sampleGap,[0.25 0.75],1);
gapRatioToPrevious=nan(1,numel(Kgrid));
relativeGapDecrease=nan(1,numel(Kgrid));
gapRatioToPrevious(2:end)=sampleGapMedian(2:end)./sampleGapMedian(1:end-1);
relativeGapDecrease(2:end)=(sampleGapMedian(1:end-1)-sampleGapMedian(2:end)) ...
    ./sampleGapMedian(1:end-1);
lastThreeRelativeDecrease=relativeGapDecrease(end-2:end);
largeSampleFlattening=all(abs(lastThreeRelativeDecrease)<0.10);
sqrtKMedianW2=sqrt(Kgrid).*(widest*medianNorm.');
logMedianW2Diagnostic=[log(Kgrid(:)),log(widest*medianNorm)];
[medianW2LogSlope,medianW2LogIntercept,medianW2LogR2]= ...
    fit_loglog_scaling(Kgrid,representativeMedianW2);
[radiusLogSlope,radiusLogIntercept,radiusLogR2]= ...
    fit_loglog_scaling(Kgrid,representativeRadius);
statisticalStudyRuntime=toc(ticStatistical);

%% Population-proxy reference
% Midpoint quadrature approximates the uniform-law population proxy.
KrefValues=[5000 10000];
nHeldoutRef=200000;
heldoutRefSeed=990001;
populationProxyX=zeros(p.N,numel(KrefValues));
populationProxyGap=zeros(numel(KrefValues),1);
populationProxyResidual=zeros(numel(KrefValues),1);
populationProxyMetrics=cell(numel(KrefValues),1);
xWarmRef=xProxyBase;
ticPopulationReference=tic;
fprintf('Computing deterministic population-proxy references...\n');
for jj=1:numel(KrefValues)
    Kref=KrefValues(jj);
    pRef=p;
    pRef.K=Kref*ones(p.N,1);
    pRef.samples=deterministic_uniform_datasets(p,Kref);
    pRef.epsilon=pBase.epsilon;
    pRef.rho=p.rhoBase;
    [xWarmRef,infoRef]=solve_proxy_ne(pRef,xWarmRef,2e-11,12000);
    metricsRef=evaluate_true_metrics( ...
        xWarmRef,pRef,p.kappaReference,nHeldoutRef,heldoutRefSeed);
    populationProxyX(:,jj)=xWarmRef;
    populationProxyGap(jj)=metricsRef.sneGap;
    populationProxyResidual(jj)=infoRef.projectedResidual;
    populationProxyMetrics{jj}=metricsRef;
    fprintf('  Kref=%d: true SNE gap %.9e, proxy residual %.3e\n', ...
        Kref,populationProxyGap(jj),populationProxyResidual(jj));
end
populationReferenceRuntime=toc(ticPopulationReference);
populationProxyEquilibrium=populationProxyX(:,end);
populationProxyReferenceGap=populationProxyGap(end);
populationReferenceXDifference=norm( ...
    populationProxyX(:,end)-populationProxyX(:,end-1));
populationReferenceGapDifference=abs( ...
    populationProxyGap(end)-populationProxyGap(end-1));
relativeK1280ReferenceDifference=abs( ...
    sampleGapMedian(end)-populationProxyReferenceGap) ...
    /max(populationProxyReferenceGap,realmin);
if max(populationProxyResidual)>2e-7
    error('DRGame:PopulationProxyResidual', ...
        'A population-proxy reference solve has excessive residual.');
end
if populationReferenceXDifference>1e-4 || ...
        populationReferenceGapDifference>1e-5
    error('DRGame:PopulationReferenceResolution', ...
        ['Kref=5000 and 10000 are not sufficiently close: ', ...
         'dx=%.3e, dg=%.3e.'],populationReferenceXDifference, ...
         populationReferenceGapDifference);
end

%% Strategic-radius sensitivity
rhoGrid=[0 0.25 0.5 0.75 1 1.05 1.25 1.5];
epsilonGrid=[0 0.25 0.5 0.75 1 1.25 1.5];
rhoRatio=rhoGrid;
rhoCoverageValid=rhoRatio>=1;
rhoExactX=zeros(p.N,numel(rhoRatio));
rhoExactGap=zeros(numel(rhoRatio),1);
rhoProxyOriginalGap=zeros(numel(rhoRatio),1);
rhoService=zeros(numel(rhoRatio),1);
xWarm=xProxyBase;
fprintf('Solving the original DRNE along the strategic-radius path...\n');
for rr=1:numel(rhoRatio)
    pc=pBase; pc.rho=rhoRatio(rr)*p.rhoCover*ones(p.N,1);
    [xWarm,info]=solve_exact_drne(pc,xWarm,2e-7,70);
    rhoExactX(:,rr)=xWarm;
    rhoExactGap(rr)=info.finalGap;
    rhoProxyOriginalGap(rr)=original_dr_nash_gap(xProxyBase,pc);
    rhoService(rr)=sum(xWarm);
end

%% Exogenous-radius sensitivity
epsilonMultiplier=epsilonGrid;
epsExactX=zeros(p.N,numel(epsilonMultiplier));
epsExactGap=zeros(numel(epsilonMultiplier),1);
epsProxyOriginalGap=zeros(numel(epsilonMultiplier),1);
epsService=zeros(numel(epsilonMultiplier),1);
xWarm=xProxyBase;
fprintf('Solving the original DRNE along the exogenous-radius path...\n');
for ee=1:numel(epsilonMultiplier)
    pc=pBase; pc.epsilon=epsilonMultiplier(ee)*pBase.epsilon;
    [xWarm,info]=solve_exact_drne(pc,xWarm,2e-7,70);
    epsExactX(:,ee)=xWarm; epsExactGap(ee)=info.finalGap;
    epsProxyOriginalGap(ee)=original_dr_nash_gap(xProxyBase,pc);
    epsService(ee)=sum(xWarm);
end

rhoBaselineIndex=find(abs(rhoRatio-1.05)<1e-12,1);
epsilonBaselineIndex=find(abs(epsilonMultiplier-1)<1e-12,1);
rhoPathBaselineGap=rhoProxyOriginalGap(rhoBaselineIndex);
epsilonPathBaselineGap=epsProxyOriginalGap(epsilonBaselineIndex);
baselinePathConsistencyError=abs( ...
    rhoPathBaselineGap-epsilonPathBaselineGap);
if baselinePathConsistencyError>1e-10
    error('DRGame:BaselinePathConsistency', ...
        'The two one-dimensional baseline gap evaluations disagree.');
end

baselineService=epsService(epsilonBaselineIndex);
baselineServiceConsistencyError=abs( ...
    baselineService-rhoService(rhoBaselineIndex));
if baselineServiceConsistencyError>1e-6
    error('DRGame:BaselineServiceConsistency', ...
        ['The exact-DRNE baseline service differs between the two paths ', ...
         'by %.3e, beyond the solver-consistent 1e-6 tolerance.'], ...
        baselineServiceConsistencyError);
end
rhoServiceRelativePercent=100*(rhoService-baselineService)/baselineService;
epsServiceRelativePercent=100*(epsService-baselineService)/baselineService;

%% Two-budget sensitivity map
% Fix the proxy equilibrium while varying the robust-game budgets.
epsilonMapGrid=[0 0.25 0.5 0.75 1 1.25 1.5];
rhoMapGrid=[0 0.25 0.5 0.75 1 1.05 1.25 1.5];
nEpsilonMap=numel(epsilonMapGrid);
nRhoMap=numel(rhoMapGrid);
budgetMapProxyGap=zeros(nRhoMap,nEpsilonMap);
budgetMapExactX=zeros(p.N,nRhoMap,nEpsilonMap);
budgetMapExactResidual=zeros(nRhoMap,nEpsilonMap);
xDRBaseline=epsExactX(:,epsilonMultiplier==1);

fprintf('Solving the 8-by-7 two-budget exact-DRNE map...\n');
xWarmMap=xDRBaseline;
for ee=1:nEpsilonMap
    if mod(ee,2)==1
        rhoOrder=1:nRhoMap;
    else
        rhoOrder=nRhoMap:-1:1;
    end
    for orderIndex=1:nRhoMap
        rr=rhoOrder(orderIndex);
        pc=pBase;
        pc.epsilon=epsilonMapGrid(ee)*pBase.epsilon;
        pc.rho=rhoMapGrid(rr)*p.rhoCover*ones(p.N,1);
        budgetMapProxyGap(rr,ee)=original_dr_nash_gap(xProxyBase,pc);
        [xWarmMap,info]=solve_exact_drne(pc,xWarmMap,3e-7,70);
        budgetMapExactX(:,rr,ee)=xWarmMap;
        budgetMapExactResidual(rr,ee)=info.finalGap;
    end
    fprintf('  exogenous multiplier %.2f complete (%d/%d columns).\n', ...
        epsilonMapGrid(ee),ee,nEpsilonMap);
end

rhoProxyDistance=vecnorm(rhoExactX-xProxyBase,2,1).';
rhoBaselineDistance=vecnorm(rhoExactX-xDRBaseline,2,1).';
epsProxyDistance=vecnorm(epsExactX-xProxyBase,2,1).';
epsBaselineDistance=vecnorm(epsExactX-xDRBaseline,2,1).';
budgetMapProxyDistance=zeros(nRhoMap,nEpsilonMap);
budgetMapBaselineDistance=zeros(nRhoMap,nEpsilonMap);
for ee=1:nEpsilonMap
    for rr=1:nRhoMap
        xMap=budgetMapExactX(:,rr,ee);
        budgetMapProxyDistance(rr,ee)=norm(xMap-xProxyBase);
        budgetMapBaselineDistance(rr,ee)=norm(xMap-xDRBaseline);
    end
end

baselineMapRhoIndex=find(abs(rhoMapGrid-1.05)<1e-12,1);
baselineMapEpsilonIndex=find(abs(epsilonMapGrid-1)<1e-12,1);
baselineMapProxyGap=budgetMapProxyGap( ...
    baselineMapRhoIndex,baselineMapEpsilonIndex);
directBaselineProxyGap=original_dr_nash_gap(xProxyBase,pBase);
baselineMapConsistencyError=abs( ...
    baselineMapProxyGap-directBaselineProxyGap);
baselineExactXConsistency=norm( ...
    budgetMapExactX(:,baselineMapRhoIndex,baselineMapEpsilonIndex) ...
    -xDRBaseline);
if baselineMapConsistencyError>1e-10
    error('DRGame:BaselineMapConsistency', ...
        'The directly evaluated and mapped baseline proxy gaps disagree.');
end

if max([rhoExactGap;epsExactGap;budgetMapExactResidual(:)])>2e-5
    error('DRGame:ExactPathResidual', ...
        'An original-game DRNE path/map solve has excessive residual.');
end

%% Dependence sensitivity at fixed marginals
% Pair random draws to isolate dependence at fixed marginal laws.
dependenceKappa=[0 0.2 0.4 0.6 0.8 1];
nDependenceRep=100;
nHeldoutDependence=20000;
dependenceHeldoutSeeds=996000+(1:nDependenceRep).';
nDependenceLevel=numel(dependenceKappa);
dependenceRepGap=nan(nDependenceRep,nDependenceLevel);
dependenceRepCorrelation=nan(nDependenceRep,nDependenceLevel);
dependenceRepStrategicUnitMean=nan(nDependenceRep,nDependenceLevel);
dependenceRepStrategicUnitVariance=nan(nDependenceRep,nDependenceLevel);
fprintf(['Running %d paired same-design dependence repetitions ', ...
    'with %d held-out samples each...\n'],nDependenceRep,nHeldoutDependence);
if useParallel
    parfor rep=1:nDependenceRep
        [dependenceRepGap(rep,:),dependenceRepCorrelation(rep,:), ...
            dependenceRepStrategicUnitMean(rep,:), ...
            dependenceRepStrategicUnitVariance(rep,:)]= ...
            evaluate_dependence_sensitivity_rep(xProxyBase,pBase, ...
            dependenceKappa,nHeldoutDependence,dependenceHeldoutSeeds(rep));
    end
else
    for rep=1:nDependenceRep
        [dependenceRepGap(rep,:),dependenceRepCorrelation(rep,:), ...
            dependenceRepStrategicUnitMean(rep,:), ...
            dependenceRepStrategicUnitVariance(rep,:)]= ...
            evaluate_dependence_sensitivity_rep(xProxyBase,pBase, ...
            dependenceKappa,nHeldoutDependence,dependenceHeldoutSeeds(rep));
    end
end
dependenceMedianGap=median(dependenceRepGap,1);
dependenceIQR=quantile(dependenceRepGap,[0.25 0.75],1);
dependenceMeanGap=mean(dependenceRepGap,1);
dependenceStdGap=std(dependenceRepGap,0,1);
dependenceEmpiricalCorrelation=mean(dependenceRepCorrelation,1);
dependenceStrategicUnitMean=mean(dependenceRepStrategicUnitMean,1);
dependenceStrategicUnitVariance=mean( ...
    dependenceRepStrategicUnitVariance,1);
dependenceMaximumCorrelationDeviation=max(abs( ...
    dependenceRepCorrelation-dependenceKappa),[],1);
dependenceMaximumMeanDeviation=max(abs( ...
    dependenceRepStrategicUnitMean-0.5),[],1);
dependenceMaximumVarianceDeviation=max(abs( ...
    dependenceRepStrategicUnitVariance-1/12),[],1);
dependenceMarginalAuditPass= ...
    dependenceMaximumMeanDeviation<5e-3 & ...
    dependenceMaximumVarianceDeviation<2e-3 & ...
    dependenceMaximumCorrelationDeviation<2e-2;
if any(~dependenceMarginalAuditPass)
    error('DRGame:DependenceMarginalAudit', ...
        ['At least one dependence level failed the paired fixed-marginal ', ...
         'mean, variance, or correlation audit.']);
end
dependenceMarginalAudit=table(dependenceKappa.', ...
    dependenceEmpiricalCorrelation.',dependenceStrategicUnitMean.', ...
    dependenceStrategicUnitVariance.', ...
    dependenceMaximumCorrelationDeviation.', ...
    dependenceMaximumMeanDeviation.', ...
    dependenceMaximumVarianceDeviation.',dependenceMarginalAuditPass.', ...
    'VariableNames',{'Kappa','EmpiricalCorrelation','StrategicUnitMean', ...
    'StrategicUnitVariance','MaxRepCorrelationDeviation', ...
    'MaxRepMeanDeviation','MaxRepVarianceDeviation','AllRepAuditsPass'});

%% Fixed-multiplier sensitivity
lambdaScaleGrid=[1 1.1 1.25 1.5 2];
nLambdaScale=numel(lambdaScaleGrid);
lambdaSweepMuZ=zeros(nLambdaScale,1);
lambdaSweepProxyX=zeros(p.N,nLambdaScale);
lambdaSweepOriginalGap=zeros(nLambdaScale,1);
lambdaSweepExactDistance=zeros(nLambdaScale,1);
lambdaSweepTauBar=zeros(nLambdaScale,1);
lambdaSweepProxyResidual=zeros(nLambdaScale,1);
lambdaSweepCertifiedMuF=zeros(nLambdaScale,1);
xWarmLambda=xProxyBase;
for ll=1:nLambdaScale
    ps=pBase;
    ps.lambdaXi=lambdaScaleGrid(ll)*p.lambdaXi;
    ps.lambdaY=lambdaScaleGrid(ll)*p.lambdaY;
    [baseScale,~]=analytical_constants_base(ps);
    ps.muZi=2*min(ps.lambdaXi,ps.lambdaY)-baseScale.ellZ-0.50;
    [~,certScale]=analytical_constants_base(ps);
    [cstScale,certScale]=analytical_constants(ps,certScale);
    [gainScale,cstScale]=select_theorem_gains(ps,cstScale,certScale);
    auditScale=theory_audit(ps,cstScale,certScale,gainScale,ps.rho);
    assert_all_pass(auditScale);
    [xWarmLambda,infoScale]=solve_proxy_ne(ps,xWarmLambda,2e-10,8000);
    lambdaSweepMuZ(ll)=min(ps.muZi);
    lambdaSweepProxyX(:,ll)=xWarmLambda;
    lambdaSweepOriginalGap(ll)=original_dr_nash_gap(xWarmLambda,pBase);
    lambdaSweepExactDistance(ll)=norm(xWarmLambda-xDRBaseline);
    lambdaSweepTauBar(ll)=gainScale.tauBar;
    lambdaSweepProxyResidual(ll)=infoScale.projectedResidual;
    lambdaSweepCertifiedMuF(ll)=certScale.muF;
end
if max(lambdaSweepProxyResidual)>2e-7
    error('DRGame:FixedDualSweepResidual', ...
        'A fixed-dual sweep proxy solve has excessive residual.');
end
fixedDualTradeoffTable=table(lambdaScaleGrid(:),lambdaSweepMuZ, ...
    lambdaSweepOriginalGap,lambdaSweepExactDistance,lambdaSweepTauBar, ...
    lambdaSweepProxyResidual,lambdaSweepCertifiedMuF, ...
    'VariableNames',{'MultiplierScale','InnerStrongConcavity', ...
    'OriginalDRGapAtProxy','DistanceToExactDRNE','TauBar', ...
    'ProxyResidual','CertifiedMuF'});

%% Figures
blue=[0.00 0.35 0.62]; red=[0.78 0.20 0.18]; green=[0.18 0.55 0.34];
neutral=[0.35 0.35 0.35];
figureResolutionDpi=600;
figureFiles=struct;
meaningfulKTicks=[5 20 80 320 1280];
coverageTarget=1-p.deltaTotal;
coverageLow=min([coverageCI(:,1);coverageTarget]);
coverageHigh=max([coverageCI(:,2);coverageTarget]);
coverageMargin=max(0.003,0.12*(coverageHigh-coverageLow));
coverageYLim=[max(0,coverageLow-coverageMargin), ...
    min(1,coverageHigh+coverageMargin)];
if diff(coverageYLim)<0.02
    coverageCenter=mean(coverageYLim);
    coverageYLim=[max(0,coverageCenter-0.01), ...
        min(1,coverageCenter+0.01)];
end
coverageTargetInsideCI=coverageCI(:,1)<=coverageTarget & ...
    coverageCI(:,2)>=coverageTarget;
coverageNoSignificantUndercoverage=coverageCI(:,2)>=coverageTarget;
xBand=[Kgrid fliplr(Kgrid)];

fig3a=publication_figure(false);
ax=axes(fig3a); hold(ax,'on');
hW2IQR=fill(ax,xBand,[representativeW2IQR(:,1).' ...
    fliplr(representativeW2IQR(:,2).')],blue, ...
    'FaceAlpha',0.15,'EdgeColor','none');
hRadius=plot(ax,Kgrid,representativeRadius,'-o','Color',blue, ...
    'MarkerFaceColor','w','MarkerSize',3.8,'LineWidth',1.50);
hW2Median=plot(ax,Kgrid,representativeMedianW2,'--s', ...
    'Color',neutral,'MarkerFaceColor','w','MarkerSize',3.4,'LineWidth',1.15);
set(ax,'XScale','log','XTick',meaningfulKTicks, ...
    'XTickLabel',string(meaningfulKTicks));
xlabel(ax,'Sample size $K$','Interpreter','latex');
ylabel(ax,'Empirical $W_2$ distance / calibrated radius','Interpreter','latex');
legend(ax,[hW2IQR hRadius hW2Median],{'IQR','radius','median'}, ...
    'Location','best');
style_axes(ax);
figureFiles.fig3a=export_standalone_figure( ...
    fig3a,outDir,'example2_fig3a',figureResolutionDpi);
close(fig3a);

fig3b=publication_figure(false);
ax=axes(fig3b); hold(ax,'on');
hCoverage=errorbar(ax,Kgrid,coverageProb,coverageProb-coverageCI(:,1), ...
    coverageCI(:,2)-coverageProb,'LineStyle','none','Marker','o', ...
    'MarkerSize',4.2,'MarkerFaceColor','w','MarkerEdgeColor',red, ...
    'Color',red,'LineWidth',1.20,'CapSize',4);
hCoverageTarget=plot(ax,[min(Kgrid) max(Kgrid)], ...
    coverageTarget*[1 1],'k--','LineWidth',1.15);
set(ax,'XScale','log','XTick',meaningfulKTicks, ...
    'XTickLabel',string(meaningfulKTicks));
ylim(ax,coverageYLim);
xlabel(ax,'Sample size $K$','Interpreter','latex');
ylabel(ax,'Empirical joint coverage','Interpreter','latex');
legend(ax,[hCoverage hCoverageTarget],{'estimate','target'}, ...
    'Location','best');
style_axes(ax);
figureFiles.fig3b=export_standalone_figure( ...
    fig3b,outDir,'example2_fig3b',figureResolutionDpi);
close(fig3b);

fig3c=publication_figure(false);
ax=axes(fig3c); hold(ax,'on');
hGapIQR=fill(ax,xBand,[sampleGapIQR(1,:) fliplr(sampleGapIQR(2,:))], ...
    green,'FaceAlpha',0.15,'EdgeColor','none');
hGapMedian=plot(ax,Kgrid,sampleGapMedian,'-o','Color',green, ...
    'MarkerFaceColor','w','MarkerSize',3.6,'LineWidth',1.55);
hPopulationReference=plot(ax,[min(Kgrid) max(Kgrid)], ...
    populationProxyReferenceGap*[1 1],'k--','LineWidth',1.10);
set(ax,'XScale','log','YScale','log','XTick',meaningfulKTicks, ...
    'XTickLabel',string(meaningfulKTicks));
xlabel(ax,'Sample size $K$','Interpreter','latex');
    ylabel(ax,'Held-out SNE-gap estimate','Interpreter','latex');
legend(ax,[hGapIQR hGapMedian hPopulationReference], ...
    {'IQR','median','population-proxy reference'},'Location','best');
style_axes(ax);
figureFiles.fig3c=export_standalone_figure( ...
    fig3c,outDir,'example2_fig3c',figureResolutionDpi);
close(fig3c);

fig4a=publication_figure(false);
ax=axes(fig4a); hold(ax,'on');
allRelative=[rhoServiceRelativePercent(:);epsServiceRelativePercent(:);0];
relativeRange=max(allRelative)-min(allRelative);
if relativeRange<=0,relativeRange=1;end
relativeYLim=[min(allRelative)-0.10*relativeRange, ...
    max(allRelative)+0.10*relativeRange];
% Mark the strategic coverage threshold only on the rho curve.
hRhoService=plot(ax,rhoRatio,rhoServiceRelativePercent,'-o', ...
    'Color',blue,'MarkerFaceColor','w','MarkerSize',3.8,'LineWidth',1.50);
hEpsService=plot(ax,epsilonMultiplier,epsServiceRelativePercent,'--s', ...
    'Color',red,'MarkerFaceColor','w','MarkerSize',3.6,'LineWidth',1.45);
hCoverageThreshold=xline(ax,1,'k:','LineWidth',1.15);
hCoverageThreshold.HandleVisibility='off';
text(ax,0.98,relativeYLim(2)-0.03*diff(relativeYLim), ...
    'strategic coverage threshold','HorizontalAlignment','right', ...
    'VerticalAlignment','top','FontSize',7.5,'Interpreter','latex', ...
    'Color',[0.15 0.15 0.15]);
plot(ax,1.05,rhoServiceRelativePercent(rhoBaselineIndex),'o', ...
    'MarkerSize',5,'MarkerFaceColor',blue,'MarkerEdgeColor',blue, ...
    'HandleVisibility','off');
plot(ax,1,epsServiceRelativePercent(epsilonBaselineIndex),'s', ...
    'MarkerSize',5,'MarkerFaceColor','w','MarkerEdgeColor',red, ...
    'LineWidth',1.0,'HandleVisibility','off');
ylim(ax,relativeYLim); xlim(ax,[0 1.5]);
xlabel(ax,'Radius multiplier $s$','Interpreter','latex');
ylabel(ax,'Relative change in total service (\%)','Interpreter','latex');
legend(ax,[hRhoService hEpsService], ...
    {'strategic radius','exogenous radius'},'Location','northoutside', ...
    'Orientation','horizontal');
style_axes(ax);
figureFiles.fig4a=export_standalone_figure( ...
    fig4a,outDir,'example2_fig4a',figureResolutionDpi);
close(fig4a);

fig4b=publication_figure(false);
ax=axes(fig4b); hold(ax,'on');
hDependenceIQR=fill(ax,[dependenceKappa fliplr(dependenceKappa)], ...
    [dependenceIQR(1,:) fliplr(dependenceIQR(2,:))],blue, ...
    'FaceAlpha',0.18,'EdgeColor','none');
hDependenceMedian=plot(ax,dependenceKappa,dependenceMedianGap,'-o', ...
    'Color',blue,'MarkerFaceColor','w','MarkerSize',4.2,'LineWidth',1.55);
dependenceYMaximum=1.10*max([dependenceIQR(:);dependenceMedianGap(:)]);
if dependenceYMaximum<=0,dependenceYMaximum=1;end
ylim(ax,[0 dependenceYMaximum]);
ax.YAxis.Exponent=0;
ytickformat(ax,'%.3f');
dependencePlotScale= ...
    'linear, zero-based to avoid exaggerating small same-design variations';
xlim(ax,[0 1]); xticks(ax,dependenceKappa);
xlabel(ax,'Dependence mixture $\kappa$','Interpreter','latex');
ylabel(ax,'Held-out SNE-gap estimate','Interpreter','latex');
legend(ax,[hDependenceIQR hDependenceMedian],{'IQR','median'}, ...
    'Location','south');
style_axes(ax);
figureFiles.fig4b=export_standalone_figure( ...
    fig4b,outDir,'example2_fig4b',figureResolutionDpi);
close(fig4b);

fig4c=publication_figure(false);
ax=axes(fig4c); hold(ax,'on');
hRhoGap=semilogy(ax,rhoRatio,max(rhoProxyOriginalGap,1e-14),'-o', ...
    'Color',blue,'MarkerFaceColor','w','MarkerSize',3.8,'LineWidth',1.50);
hEpsilonGap=semilogy(ax,epsilonMultiplier,max(epsProxyOriginalGap,1e-14),'--s', ...
    'Color',red,'MarkerFaceColor','w','MarkerSize',3.6,'LineWidth',1.45);
plot(ax,1.05,rhoPathBaselineGap,'o','MarkerSize',5, ...
    'MarkerFaceColor',blue,'MarkerEdgeColor',blue,'HandleVisibility','off');
plot(ax,1,epsilonPathBaselineGap,'s','MarkerSize',5, ...
    'MarkerFaceColor','w','MarkerEdgeColor',red,'LineWidth',1.0, ...
    'HandleVisibility','off');
set(ax,'YScale','log'); xlim(ax,[0 1.5]);
xlabel(ax,'One-at-a-time radius multiplier $s$','Interpreter','latex');
ylabel(ax,'$g_{\rm DR}(x_{\rm proxy})$','Interpreter','latex');
legend(ax,[hRhoGap hEpsilonGap],{'strategic radius','exogenous radius'}, ...
    'Location','best');
style_axes(ax);
figureFiles.fig4c=export_standalone_figure( ...
    fig4c,outDir,'example2_fig4c',figureResolutionDpi);
close(fig4c);

fig4d=publication_figure(true);
ax=axes(fig4d); hold(ax,'on');
mapGapMin=min(budgetMapProxyGap,[],'all');
mapGapMax=max(budgetMapProxyGap,[],'all');
mapColorLimits=[mapGapMin mapGapMax];
draw_discrete_heatmap(ax,epsilonMapGrid,rhoMapGrid,budgetMapProxyGap);
clim(ax,mapColorLimits);
patch(ax,[min(epsilonMapGrid) max(epsilonMapGrid) ...
    max(epsilonMapGrid) min(epsilonMapGrid)], ...
    [min(rhoMapGrid) min(rhoMapGrid) 1 1],[0.92 0.92 0.92], ...
    'FaceAlpha',0.28,'EdgeColor','none','HandleVisibility','off');
yline(ax,1,'k--','LineWidth',1.05,'HandleVisibility','off');
plot(ax,1,1.05,'p','MarkerSize',9,'MarkerFaceColor','w', ...
    'MarkerEdgeColor','k','LineWidth',1.0,'HandleVisibility','off');
xlim(ax,[min(epsilonMapGrid) max(epsilonMapGrid)]);
ylim(ax,[min(rhoMapGrid) max(rhoMapGrid)]);
xlabel(ax,'Exogenous-radius multiplier $s_\varepsilon$','Interpreter','latex');
ylabel(ax,'Strategic-radius multiplier $s_\rho$','Interpreter','latex');
cb=colorbar(ax);
cb.Label.String='$g_{\rm DR}(x_{\rm proxy})$';
cb.Label.Interpreter='latex';
cb.FontName='Times New Roman'; cb.FontSize=8;
mapN=160;
blueMap=[linspace(0.96,0.05,mapN).', ...
    linspace(0.97,0.32,mapN).',linspace(0.99,0.55,mapN).'];
colormap(fig4d,blueMap);
style_axes(ax);
figureFiles.fig4d=export_standalone_figure( ...
    fig4d,outDir,'example2_fig4d',figureResolutionDpi);
close(fig4d);

%% Save results
maximumExactResidual=max([rhoExactGap;epsExactGap;budgetMapExactResidual(:)]);
publicationAuditSummary=struct( ...
    'representativeRadiusAtSmallestK',representativeRadius(1), ...
    'representativeRadiusAtLargestK',representativeRadius(end), ...
    'coverageMinimum',min(coverageProb), ...
    'coverageMean',mean(coverageProb), ...
    'coverageMaximum',max(coverageProb), ...
    'coverageTarget',coverageTarget, ...
    'medianW2LogLogSlope',medianW2LogSlope, ...
    'medianW2LogLogR2',medianW2LogR2, ...
    'radiusLogLogSlope',radiusLogSlope, ...
    'radiusLogLogR2',radiusLogR2, ...
    'heldoutGapAtSmallestK',sampleGapMedian(1), ...
    'heldoutGapAtLargestK',sampleGapMedian(end), ...
    'populationProxyReferenceGap',populationProxyReferenceGap, ...
    'strategicServiceRelativeRange',[min(rhoServiceRelativePercent) ...
        max(rhoServiceRelativePercent)], ...
    'exogenousServiceRelativeRange',[min(epsServiceRelativePercent) ...
        max(epsServiceRelativePercent)], ...
    'baselineProxyOriginalGameGap',directBaselineProxyGap, ...
    'maximumExactDRNEResidual',maximumExactResidual);
results=struct;
results.parameters=p;
results.publicationAuditSummary=publicationAuditSummary;
results.Kgrid=Kgrid;
results.nCalibration=nCalibration;
results.nCoverage=nCoverage;
results.nEquilibriumRep=nEquilibriumRep;
results.nHeldout=nHeldout;
results.useParallel=useParallel;
results.parallelWorkers=parallelWorkers;
results.calibrationSeeds=calibrationSeeds;
results.coverageSeeds=coverageSeeds;
results.equilibriumDatasetSeeds=equilibriumDatasetSeeds;
results.heldoutSeeds=heldoutSeeds;
results.empiricalRadius=epsilonByK;
results.representativePlayer=representativePlayer;
results.representativeSupportWidth=supportWidth(representativePlayer);
results.representativePlayerRadius=representativeRadius;
results.maxPlayerRadius=maxPlayerRadius;
results.representativePlayerW2Quantiles=representativeW2Quantiles;
results.maxPlayerW2Quantiles=maxPlayerW2Quantiles;
results.representativePlayerMedianW2=representativeMedianW2;
results.representativePlayerW2IQR=representativeW2IQR;
results.maxPlayerMedianW2=maxPlayerMedianW2;
results.maxPlayerW2IQR=maxPlayerW2IQR;
results.normalizedSelectedW2Radius=qNorm;
results.normalizedMedianW2=medianNorm;
results.normalizedW2IQR=iqrNorm;
results.w2QuantileLevels=quantileLevels;
results.normalizedW2Quantiles=w2QuantileNorm;
results.normalizedW2QuantilesByPlayer=w2QuantileNormByPlayer;
results.maxSupportW2Quantiles=w2QuantileMaxSupport;
results.playerW2Quantiles=w2QuantileByPlayer;
results.widestSupportPlayer=widestPlayer;
results.normalizedW2CalibrationReplicates=w2CalibrationNorm;
results.normalizedW2CoverageReplicates=coverageW2Norm;
results.jointCoverageIndicators=coverageIndicators;
results.coverageProbability=coverageProb;
results.coverageWilsonCI=coverageCI;
results.coverageTarget=coverageTarget;
results.coverageTargetInsideWilsonCI=coverageTargetInsideCI;
results.coverageNoSignificantUndercoverage=coverageNoSignificantUndercoverage;
results.coveragePlotYLim=coverageYLim;
results.sampleGap=sampleGap;
results.sampleGapMedian=sampleGapMedian;
results.sampleGapIQR=sampleGapIQR;
results.sampleGapRatioToPrevious=gapRatioToPrevious;
results.sampleGapRelativeDecrease=relativeGapDecrease;
results.lastThreeRelativeGapDecrease=lastThreeRelativeDecrease;
results.largeSampleFlattening=largeSampleFlattening;
results.largeSampleFlatteningRule= ...
    'absolute median-gap decrease below 10 percent in each of the last three K doublings';
results.sqrtKMedianW2=sqrtKMedianW2;
results.logMedianW2Diagnostic=logMedianW2Diagnostic;
results.representativeMedianW2LogLogFit=struct( ...
    'slope',medianW2LogSlope,'intercept',medianW2LogIntercept, ...
    'R2',medianW2LogR2,'interpretation','empirical diagnostic, not a sharp theorem rate');
results.representativeRadiusLogLogFit=struct( ...
    'slope',radiusLogSlope,'intercept',radiusLogIntercept, ...
    'R2',radiusLogR2,'interpretation','empirical diagnostic, not a sharp theorem rate');
results.statisticalStudyRuntimeSeconds=statisticalStudyRuntime;
results.sampleX=sampleX;
results.sampleProxyResidual=sampleResidual;
results.sampleService=sampleService;
results.sampleRisk=sampleRisk;
results.KrefValues=KrefValues;
results.nHeldoutRef=nHeldoutRef;
results.heldoutRefSeed=heldoutRefSeed;
results.populationProxyX=populationProxyX;
results.populationProxyGap=populationProxyGap;
results.populationProxyResidual=populationProxyResidual;
results.populationProxyMetrics=populationProxyMetrics;
results.populationProxyEquilibrium=populationProxyEquilibrium;
results.populationProxyReferenceGap=populationProxyReferenceGap;
results.populationReferenceXDifference=populationReferenceXDifference;
results.populationReferenceGapDifference=populationReferenceGapDifference;
results.relativeK1280ReferenceDifference=relativeK1280ReferenceDifference;
results.populationReferenceRuntimeSeconds=populationReferenceRuntime;
results.theoryByK=theoryByK;
results.gainByK=gainByK;
results.auditByK=auditByK;
results.certifiedMuF=cert.muF;
results.numericMuObserved=numericMuObserved;
results.xProxyBase=xProxyBase;
results.proxyBaseInfo=proxyBaseInfo;
results.baseDatasetSeed=baseDatasetSeed;
results.baseSamples=baseSamples;
results.trueLawKappaCorr=p.kappaReference;
results.trueLawKappaMixture=p.kappaReference;
results.trueLawConstruction= ...
    'V=U with probability kappa and V=U_j otherwise; both marginals remain Uniform(0,1)';
results.nonProductTrueJointLaw=true;
results.sGrid=epsilonGrid;
results.rhoGrid=rhoGrid;
results.epsilonGrid=epsilonGrid;
results.rhoRatio=rhoRatio;
results.rhoCoverageValid=rhoCoverageValid;
results.rhoExactX=rhoExactX;
results.rhoExactGap=rhoExactGap;
results.rhoProxyOriginalGap=rhoProxyOriginalGap;
results.rhoService=rhoService;
results.rhoServiceRelativePercent=rhoServiceRelativePercent;
results.rhoExactToProxyDistance=rhoProxyDistance;
results.rhoExactToBaselineDistance=rhoBaselineDistance;
results.epsilonMultiplier=epsilonMultiplier;
results.epsExactX=epsExactX;
results.epsExactGap=epsExactGap;
results.epsProxyOriginalGap=epsProxyOriginalGap;
results.epsService=epsService;
results.epsServiceRelativePercent=epsServiceRelativePercent;
results.epsExactToProxyDistance=epsProxyDistance;
results.epsExactToBaselineDistance=epsBaselineDistance;
results.rhoPathBaselineGap=rhoPathBaselineGap;
results.epsilonPathBaselineGap=epsilonPathBaselineGap;
results.baselinePathConsistencyError=baselinePathConsistencyError;
results.baselineService=baselineService;
results.baselineServiceConsistencyError=baselineServiceConsistencyError;
results.mapGrid=epsilonMapGrid;
results.epsilonMapGrid=epsilonMapGrid;
results.rhoMapGrid=rhoMapGrid;
results.budgetMapProxyGap=budgetMapProxyGap;
results.budgetMapExactX=budgetMapExactX;
results.budgetMapExactResidual=budgetMapExactResidual;
results.budgetMapExactToProxyDistance=budgetMapProxyDistance;
results.budgetMapExactToBaselineDistance=budgetMapBaselineDistance;
results.mapColorLimits=mapColorLimits;
results.mapRepresentation='discrete exact-grid cells; no interpolated contour surface';
results.xExactDRBaseline=xDRBaseline;
results.baselineMapProxyGap=baselineMapProxyGap;
results.directBaselineProxyGap=directBaselineProxyGap;
results.baselineMapConsistencyError=baselineMapConsistencyError;
results.baselineExactXConsistency=baselineExactXConsistency;
results.dependenceKappa=dependenceKappa;
results.dependenceFixedDesign=xProxyBase;
results.dependenceFixedDesignProxyResidual=proxyBaseInfo.projectedResidual;
results.dependenceBaselineK=Kbase;
results.dependenceFixedEpsilon=pBase.epsilon;
results.dependenceFixedRho=pBase.rho;
results.dependenceFixedLambdaXi=pBase.lambdaXi;
results.dependenceFixedLambdaY=pBase.lambdaY;
results.nDependenceRep=nDependenceRep;
results.nHeldoutDependence=nHeldoutDependence;
results.dependenceHeldoutSeeds=dependenceHeldoutSeeds;
results.dependenceRepGap=dependenceRepGap;
results.dependenceMedianGap=dependenceMedianGap;
results.dependenceIQR=dependenceIQR;
results.dependenceMeanGap=dependenceMeanGap;
results.dependenceStdGap=dependenceStdGap;
results.dependenceEmpiricalCorrelation=dependenceEmpiricalCorrelation;
results.dependenceStrategicUnitMean=dependenceStrategicUnitMean;
results.dependenceStrategicUnitVariance=dependenceStrategicUnitVariance;
results.dependenceRepCorrelation=dependenceRepCorrelation;
results.dependenceRepStrategicUnitMean=dependenceRepStrategicUnitMean;
results.dependenceRepStrategicUnitVariance= ...
    dependenceRepStrategicUnitVariance;
results.dependenceMarginalAudit=dependenceMarginalAudit;
results.dependenceMarginalAuditPass=dependenceMarginalAuditPass;
results.dependencePairedCommonRandomNumbers=true;
results.dependencePlotScale=dependencePlotScale;
results.dependenceStudyDefinition=[ ...
    'Sensitivity of the baseline proxy equilibrium to exogenous-strategic ', ...
    'dependence. Marginal distributions are preserved and only their ', ...
    'dependence is varied; no equilibrium is recomputed across kappa.'];
results.fig4aCoverageThresholdMeaning=[ ...
    's=1 is the strategic-radius coverage threshold only; it does not ', ...
    'apply to the exogenous-radius curve.'];
results.fig4cInterpretation=[ ...
    'The proxy equilibrium is fixed while the original robust game changes ', ...
    'with one radius at a time; nonmonotonicity is not smoothed.'];
results.fixedDualTradeoff=fixedDualTradeoffTable;
results.fixedDualLambdaScaleGrid=lambdaScaleGrid;
results.fixedDualProxyX=lambdaSweepProxyX;
results.figureFiles=figureFiles;
results.figureResolutionDpi=figureResolutionDpi;
results.meaningfulKPlotTicks=meaningfulKTicks;
save(fullfile(outDir,'example2_results.mat'),'results','-v7.3');

fprintf('\nExample 2 completed.\n');
fprintf('Sample sizes: %s\n', mat2str(Kgrid));
fprintf('Joint coverage range / target: %.4f to %.4f / %.4f\n', ...
    min(coverageProb), max(coverageProb), coverageTarget);
fprintf('Median held-out SNE gap at K=%d / K=%d: %.6e / %.6e\n', ...
    Kgrid(1), Kgrid(end), sampleGapMedian(1), sampleGapMedian(end));
fprintf('Population-proxy reference gap: %.6e\n', populationProxyReferenceGap);
fprintf('Maximum reference-DRNE residual: %.3e\n', maximumExactResidual);
fprintf('Results saved to: %s\n', outDir);

%% Local functions
function d=uniform_w2_replicates(K,nRep,seed)
stream=RandStream('mt19937ar','Seed',seed);
ordered=sort(rand(stream,K,nRep),1);
u0=(0:K-1).'/K; u1=(1:K).'/K;
deltaU2=u1.^2-u0.^2; deltaU3=u1.^3-u0.^3;
w2Squared=sum(ordered.^2/K-ordered.*deltaU2+deltaU3/3,1);
d=sqrt(max(w2Squared,0)).';
end

function data=generate_datasets(p,K,seed)
stream=RandStream('mt19937ar','Seed',seed); data=cell(p.N,1);
for i=1:p.N
    data{i}=p.wL(i)+(p.wU(i)-p.wL(i))*rand(stream,K,1);
end
end

function data=deterministic_uniform_datasets(p,K)
u=((1:K).'-0.5)/K;
data=cell(p.N,1);
for i=1:p.N
    data{i}=p.wL(i)+(p.wU(i)-p.wL(i))*u;
end
end

function [x,gap,service,risk,residual]=run_sample_replication( ...
    p,K,epsilon,nHeldout,datasetSeed,heldoutSeed)
pRep=p;
pRep.K=K*ones(p.N,1);
pRep.samples=generate_datasets(p,K,datasetSeed);
pRep.epsilon=epsilon;
pRep.rho=p.rhoBase;
[x,info]=solve_proxy_ne(pRep,nominal_ne(pRep),2e-10,8000);
metrics=evaluate_true_metrics(x,pRep,p.kappaReference,nHeldout,heldoutSeed);
gap=metrics.sneGap;
service=metrics.totalService;
risk=metrics.congestionQ95;
residual=info.projectedResidual;
end

function [cst,cert]=analytical_constants_base(p)
cst.ellZ=zeros(p.N,1); cst.ellJ=zeros(p.N,1);
cert.crossLipschitz=zeros(p.N); cert.FcomponentLipschitz=zeros(p.N);
for i=1:p.N
    b=p.B(i,:).'; nb=norm(b); bsum=sum(b);
    cst.ellZ(i)=p.c(i)*nb;
    mixed=max(p.r(i),abs(p.c(i)*bsum-p.r(i)));
    cst.ellJ(i)=sqrt(p.a(i)^2+2*mixed^2+ ...
        2*p.c(i)^2*(p.wU(i)^2+1)*nb^2);
end
if ~isfield(p,'muZi'), cert.muF=nan; cert.ellF=nan; return; end
for i=1:p.N
    b=p.B(i,:).'; nb=norm(b); mixed=max(p.r(i),abs(p.c(i)*sum(b)-p.r(i)));
    mZeta=2*p.lambdaXi(i)-p.c(i)^2*nb^2/(2*p.lambdaY(i));
    partialFz=mixed+p.wU(i)*p.c(i)^2*nb^2/(2*p.lambdaY(i));
    gradFz=sqrt(mixed^2+(p.c(i)*p.wU(i)*nb)^2);
    cert.FcomponentLipschitz(i,i)=p.a(i)+gradFz^2/p.muZi(i);
    for j=1:p.N
        if j~=i && p.B(i,j)~=0
            Lij=p.c(i)*p.B(i,j)*(p.wU(i)+partialFz/mZeta);
            cert.crossLipschitz(i,j)=Lij;
            cert.FcomponentLipschitz(i,j)=Lij;
        end
    end
end
muRows=zeros(p.N,1);
for i=1:p.N
    muRows(i)=p.a(i)-0.5*sum(cert.crossLipschitz(i,:) ...
        +cert.crossLipschitz(:,i).');
end
cert.muRows=muRows; cert.muF=min(muRows);
cert.ellF=norm(cert.FcomponentLipschitz,2);
end

function [cst,cert]=analytical_constants(p,cert)
[base,~]=analytical_constants_base(p);
cst=base; cst.muZ=min(p.muZi);
cst.ellXzI=cst.ellJ+2*p.lambdaY;
cst.ellGx=max(cst.ellJ); cst.ellGz=max(cst.ellJ./sqrt(p.K));
cst.ellHx=max(sqrt(p.K).*cst.ellXzI);
cst.ellHz=max(cst.ellJ+2*max(p.lambdaXi,p.lambdaY));
cst.ellH=cst.ellHx/cst.muZ;
cst.ellFtilde=cst.ellGx+cst.ellGz*cst.ellH;
cst.ellF=cert.ellF;
end

function [gain,cst]=select_theorem_gains(p,cst,cert)
gain.gammaMin=(cst.ellFtilde+(cst.ellF+cst.ellFtilde)^2/(4*cert.muF))/p.lambda2;
best=-inf;
for factor=linspace(1.001,8,3000)
    gamma=factor*gain.gammaMin;
    Q=[cert.muF/p.N,-(cst.ellF+cst.ellFtilde)/(2*sqrt(p.N)); ...
      -(cst.ellF+cst.ellFtilde)/(2*sqrt(p.N)),gamma*p.lambda2-cst.ellFtilde];
    mu=min(eig((Q+Q.')/2)); ell=cst.ellFtilde+gamma*p.lambdaN;
    if mu>0 && mu/ell>best
        best=mu/ell; gain.gamma=gamma; gain.Qgamma=Q;
        cst.muGamma=mu; cst.ellGamma=ell;
    end
end
gain.alphaMax=2*cst.muGamma/cst.ellGamma^2;
gain.betaMax=2*cst.muZ/cst.ellHz^2;
gain.alpha=cst.muGamma/cst.ellGamma^2;
gain.beta=cst.muZ/cst.ellHz^2;
gain.qx=sqrt(max(0,1-2*gain.alpha*cst.muGamma+gain.alpha^2*cst.ellGamma^2));
gain.qz=sqrt(max(0,1-2*gain.beta*cst.muZ+gain.beta^2*cst.ellHz^2));
omx=(2*gain.alpha*cst.muGamma-gain.alpha^2*cst.ellGamma^2)/(1+gain.qx);
omz=(2*gain.beta*cst.muZ-gain.beta^2*cst.ellHz^2)/(1+gain.qz);
gain.tauBar=omx*omz/(2*gain.alpha*cst.ellH*cst.ellGz);
gain.tau=0.35*gain.tauBar;
kappa=gain.alpha*cst.ellGz/(cst.ellH*(1+gain.qx));
Theta=[omx,-gain.alpha*cst.ellGz; -gain.alpha*cst.ellGz, ...
    kappa*(omz/gain.tau-gain.alpha*cst.ellH*cst.ellGz)];
cst.thetaMin=min(eig((Theta+Theta.')/2));
end

function audit=theory_audit(p,cst,cert,gain,rhoSelected)
condition=["compact X";"own curvature";"finite Lipschitz bounds"; ...
    "inner strong concavity";"proxy strong monotonicity";"connected graph"; ...
    "gamma lower bound";"Q_gamma positive definite";"alpha interval";"beta interval"; ...
    "q_x contraction";"q_z contraction";"tau interval"; ...
    "Theta_tau positive definite";"strategic coverage"];
margin=[1;min(p.a);min(cst.ellJ); ...
    min(2*min(p.lambdaXi,p.lambdaY)-cst.ellZ-p.muZi);cert.muF;p.lambda2; ...
    gain.gamma-gain.gammaMin;cst.muGamma;min(gain.alpha,gain.alphaMax-gain.alpha); ...
    min(gain.beta,gain.betaMax-gain.beta);1-gain.qx;1-gain.qz; ...
    min(gain.tau,gain.tauBar-gain.tau);cst.thetaMin; ...
    min(rhoSelected-p.rhoCover)];
pass=margin>0; pass(end)=margin(end)>=-1e-14;
audit=table(condition,margin,pass,'VariableNames',{'Condition','StrictMargin','Pass'});
end

function assert_all_pass(audit)
if any(~audit.Pass), disp(audit(~audit.Pass,:)); error('DRGame:TheoryAuditFailed','Mandatory condition failed.'); end
end

function [minimumEigenvalue,nPoints]=numerical_monotonicity_audit(p,nRandom)
rng(7751,'twister'); X=[rand(nRandom,p.N);zeros(1,p.N);ones(1,p.N);0.5*ones(1,p.N)];
minimumEigenvalue=inf; h=2e-6;
for q=1:size(X,1)
    x=X(q,:).'; J=zeros(p.N);
    for j=1:p.N
        xp=x;xm=x;xp(j)=min(1,x(j)+h);xm(j)=max(0,x(j)-h);
        J(:,j)=(proxy_pseudogradient(xp,p)-proxy_pseudogradient(xm,p))/(xp(j)-xm(j));
    end
    minimumEigenvalue=min(minimumEigenvalue,min(eig((J+J.')/2)));
end
nPoints=size(X,1);
end

function x=nominal_ne(p)
x=0.5*ones(p.N,1);
for it=1:3000
    old=x;
    for i=1:p.N
        oi=[1:i-1,i+1:p.N]; mw=mean(p.samples{i});
        coeff=mw*(-p.r(i)+p.c(i)*p.B(i,oi)*x(oi));
        x(i)=min(1,max(0,-coeff/p.a(i)));
    end
    if norm(x-old,inf)<1e-13,break;end
end
end

function [x,info]=solve_proxy_ne(p,x0,tol,maxIter)
x=min(1,max(0,x0(:))); step=0.14;
for it=1:maxIter
    F=proxy_pseudogradient(x,p); xn=min(1,max(0,x-step*F));
    if norm(xn-x,inf)<tol,x=xn;break;end
    x=xn;
end
info.iterations=it;
info.projectedResidual=norm(x-min(1,max(0,x-proxy_pseudogradient(x,p))),inf);
end

function F=proxy_pseudogradient(x,p)
F=zeros(p.N,1);
for i=1:p.N
    oi=[1:i-1,i+1:p.N];
    [~,z,v]=inner_max_batch(x(i),x(oi),p.samples{i},p.lambdaXi(i),p.lambdaY(i),i,p);
    F(i)=mean(p.a(i)*x(i)-p.r(i)*z+p.c(i)*z.*(v*p.B(i,oi).'));
end
end

function [x,info]=solve_exact_drne(p,x0,tol,maxSweeps)
x=min(1,max(0,x0(:))); warm=[p.lambdaXi,p.lambdaY]; relax=0.88;
for sweep=1:maxSweeps
    old=x;
    for i=1:p.N
        [br,lam]=exact_player_best_response(i,x,p,warm(i,:));
        x(i)=relax*br+(1-relax)*x(i); warm(i,:)=lam;
    end
    if norm(x-old,inf)<tol,break;end
end
info.sweeps=sweep; info.lambda=warm;
info.finalGap=original_dr_nash_gap(x,p);
end

function gap=original_dr_nash_gap(x,p)
g=zeros(p.N,1);
for i=1:p.N
    current=exact_robust_cost_i(x(i),x,i,p,[p.lambdaXi(i),p.lambdaY(i)]);
    [~,~,best]=exact_player_best_response(i,x,p,[p.lambdaXi(i),p.lambdaY(i)]);
    g(i)=max(0,current-best);
end
gap=max(g);
end

function [bestX,bestLambda,bestValue]=exact_player_best_response(i,x,p,lambda0)
oi=[1:i-1,i+1:p.N]; epsVal=p.epsilon(i);rhoVal=p.rho(i);
tol=1e-13; ubLam=2000;
opts=optimoptions('fmincon','Display','off','Algorithm','sqp', ...
    'SpecifyObjectiveGradient',true,'OptimalityTolerance',5e-10, ...
    'StepTolerance',5e-11,'ConstraintTolerance',1e-11, ...
    'MaxIterations',220,'MaxFunctionEvaluations',6000);
if epsVal>tol && rhoVal>tol
    u0=[x(i);max(lambda0(:),1e-3)]; lb=[0;0;0];ub=[1;ubLam;ubLam]; mode=3;
elseif epsVal<=tol && rhoVal>tol
    u0=[x(i);max(lambda0(2),1e-3)];lb=[0;0];ub=[1;ubLam];mode=2;
elseif epsVal>tol && rhoVal<=tol
    u0=[x(i);max(lambda0(1),1e-3)];lb=[0;0];ub=[1;ubLam];mode=1;
else
    mw=mean(p.samples{i}); coeff=mw*(-p.r(i)+p.c(i)*p.B(i,oi)*x(oi));
    bestX=min(1,max(0,-coeff/p.a(i)));bestLambda=[nan nan];
    bestValue=0.5*p.a(i)*bestX^2+bestX*coeff;return;
end
fun=@(u) player_u_objective(u,x(oi),p.samples{i},epsVal,rhoVal,i,p,mode);
[u,bestValue]=fmincon(fun,u0,[],[],[],[],lb,ub,[],opts);
bestX=u(1);
if mode==3,bestLambda=u(2:3).';elseif mode==2,bestLambda=[0 u(2)];else,bestLambda=[u(2) 0];end
if any(bestLambda(isfinite(bestLambda))>0.95*ubLam),error('DRGame:DualBound','Dual bound hit.');end
end

function value=exact_robust_cost_i(xi,x,i,p,lambda0)
oi=[1:i-1,i+1:p.N];epsVal=p.epsilon(i);rhoVal=p.rho(i);tol=1e-13;ubLam=2000;
opts=optimoptions('fmincon','Display','off','Algorithm','sqp', ...
    'SpecifyObjectiveGradient',true,'OptimalityTolerance',5e-10, ...
    'StepTolerance',5e-11,'ConstraintTolerance',1e-11, ...
    'MaxIterations',180,'MaxFunctionEvaluations',5000);
if epsVal>tol && rhoVal>tol
    mode=3;u0=max(lambda0(:),1e-3);lb=[0;0];ub=[ubLam;ubLam];
elseif epsVal<=tol && rhoVal>tol
    mode=2;u0=max(lambda0(2),1e-3);lb=0;ub=ubLam;
elseif epsVal>tol && rhoVal<=tol
    mode=1;u0=max(lambda0(1),1e-3);lb=0;ub=ubLam;
else
    mw=mean(p.samples{i});value=0.5*p.a(i)*xi^2+ ...
        xi*mw*(-p.r(i)+p.c(i)*p.B(i,oi)*x(oi));return;
end
fun=@(u) lambda_u_objective(u,xi,x(oi),p.samples{i},epsVal,rhoVal,i,p,mode);
[~,value]=fmincon(fun,u0,[],[],[],[],lb,ub,[],opts);
end

function [f,grad]=player_u_objective(u,xminus,samples,epsVal,rhoVal,i,p,mode)
if mode==3,lambda=u(2:3);elseif mode==2,lambda=[0;u(2)];else,lambda=[u(2);0];end
[f,gx,gl]=semidual_eval_general(u(1),xminus,samples,epsVal,rhoVal,lambda,i,p,mode);
if mode==3,grad=[gx;gl];elseif mode==2,grad=[gx;gl(2)];else,grad=[gx;gl(1)];end
end

function [f,grad]=lambda_u_objective(u,xi,xminus,samples,epsVal,rhoVal,i,p,mode)
if mode==3,lambda=u(:);elseif mode==2,lambda=[0;u];else,lambda=[u;0];end
[f,~,gl]=semidual_eval_general(xi,xminus,samples,epsVal,rhoVal,lambda,i,p,mode);
if mode==3,grad=gl;elseif mode==2,grad=gl(2);else,grad=gl(1);end
end

function [f,gx,gl]=semidual_eval_general(xi,xminus,samples,epsVal,rhoVal,lambda,i,p,mode)
b=p.B(i,[1:i-1,i+1:p.N]).';xminus=xminus(:);
if mode==3
    [vals,z,v]=inner_max_batch(xi,xminus,samples,lambda(1),lambda(2),i,p);
elseif mode==2
    z=samples(:);v=adversarial_v(xi,xminus.',z,lambda(2),b.',p.c(i));
    vals=0.5*p.a(i)*xi^2-p.r(i)*xi*z+p.c(i)*xi*z.*(v*b) ...
        -lambda(2)*sum((v-xminus.').^2,2);
else
    v=repmat(xminus.',numel(samples),1);
    [vals,z]=inner_z_fixed_v(xi,xminus,samples,lambda(1),i,p);
end
f=lambda(1)*epsVal^2+lambda(2)*rhoVal^2+mean(vals);
gx=mean(p.a(i)*xi-p.r(i)*z+p.c(i)*z.*(v*b));
gl=[epsVal^2-mean((z-samples(:)).^2); ...
    rhoVal^2-mean(sum((v-xminus.').^2,2))];
end

function [vals,z]=inner_z_fixed_v(xi,xminus,samples,lambdaXi,i,p)
b=p.B(i,[1:i-1,i+1:p.N]).';q=xi*(-p.r(i)+p.c(i)*b.'*xminus);
if lambdaXi>1e-13,z=samples(:)+q/(2*lambdaXi);z=min(p.wU(i),max(p.wL(i),z));
elseif q>=0,z=p.wU(i)*ones(numel(samples),1);else,z=p.wL(i)*ones(numel(samples),1);end
vals=0.5*p.a(i)*xi^2+q*z-lambdaXi*(z-samples(:)).^2;
end

function [values,zetaBest,vBest]=inner_max_batch(xi,xminus,samples,lambdaXi,lambdaY,i,p)
xminus=xminus(:).';samples=samples(:);b=p.B(i,[1:i-1,i+1:p.N]);
K=numel(samples);d=numel(b);wL=p.wL(i);wU=p.wU(i);breaks=[wL;wU];
if lambdaY>1e-13 && xi>1e-13
    qc=xi*p.c(i)*b;
    for j=1:d
        if qc(j)>0
            zb=2*lambdaY*(1-xminus(j))/qc(j);
            if zb>wL && zb<wU,breaks(end+1,1)=zb;end %#ok<AGROW>
        end
    end
end
breaks=unique(sort(breaks));best=-inf(K,1);zetaBest=wL*ones(K,1);
for seg=1:numel(breaks)-1
    left=breaks(seg);right=breaks(seg+1);mid=0.5*(left+right);
    if lambdaY>1e-13
        vm=min(1,max(0,xminus+xi*p.c(i)*mid*b/(2*lambdaY)));
        unsat=vm<1-1e-10 & vm>1e-10;
    else
        unsat=false(1,d);
    end
    A=-lambdaXi;Bbase=-p.r(i)*xi;
    for j=1:d
        qj=xi*p.c(i)*b(j);
        if lambdaY<=1e-13
            if qj>=0,Bbase=Bbase+qj;end
        elseif unsat(j)
            A=A+qj^2/(4*lambdaY);Bbase=Bbase+qj*xminus(j);
        else
            Bbase=Bbase+qj;
        end
    end
    Bvec=Bbase+2*lambdaXi*samples;cand=[left*ones(K,1),right*ones(K,1)];
    if A<-1e-14,cand(:,3)=min(right,max(left,-Bvec/(2*A)));end
    for cc=1:size(cand,2)
        zc=cand(:,cc);vc=adversarial_v(xi,xminus,zc,lambdaY,b,p.c(i));
        val=0.5*p.a(i)*xi^2-p.r(i)*xi*zc+p.c(i)*xi*zc.*(vc*b.') ...
            -lambdaXi*(zc-samples).^2-lambdaY*sum((vc-xminus).^2,2);
        take=val>best;best(take)=val(take);zetaBest(take)=zc(take);
    end
end
vBest=adversarial_v(xi,xminus,zetaBest,lambdaY,b,p.c(i));values=best;
end

function v=adversarial_v(xi,xminus,zeta,lambdaY,b,ci)
K=numel(zeta);
if lambdaY>1e-13
    v=min(1,max(0,repmat(xminus,K,1)+(xi*ci/(2*lambdaY))*(zeta(:)*b)));
else
    v=repmat(xminus,K,1);positive=xi*ci*(zeta(:)*b)>=0;v(positive)=1;
end
end

function [gapByKappa,correlationByKappa,meanByKappa,varianceByKappa]= ...
    evaluate_dependence_sensitivity_rep(x,p,kappaGrid,nHeldout,seed)
% Reuse random draws across kappa values for paired estimates.
stream=RandStream('mt19937ar','Seed',seed);
nKappa=numel(kappaGrid);
playerGap=zeros(p.N,nKappa);
rawCorrelation=zeros(p.N,nKappa);
strategicUnitMean=zeros(p.N,nKappa);
strategicUnitVariance=zeros(p.N,nKappa);
for i=1:p.N
    oi=[1:i-1,i+1:p.N];
    b=p.B(i,oi).';
    exogenousUnit=rand(stream,nHeldout,1);
    independentStrategicUnit=rand(stream,nHeldout,p.N-1);
    mixtureSelectorUniform=rand(stream,nHeldout,p.N-1);
    exogenousMatrix=repmat(exogenousUnit,1,p.N-1);
    w=p.wL(i)+(p.wU(i)-p.wL(i))*exogenousUnit;
    for dd=1:nKappa
        strategicUnit=independentStrategicUnit;
        mixtureSelector=mixtureSelectorUniform<kappaGrid(dd);
        strategicUnit(mixtureSelector)=exogenousMatrix(mixtureSelector);
        dev=p.sigmaY*(2*strategicUnit-1);
        Y=min(1,max(0,x(oi).'+dev));
        wy=mean(w.*Y,1).';
        coeff=-p.r(i)*mean(w)+p.c(i)*b.'*wy;
        br=min(1,max(0,-coeff/p.a(i)));
        current=0.5*p.a(i)*x(i)^2+coeff*x(i);
        best=0.5*p.a(i)*br^2+coeff*br;
        playerGap(i,dd)=max(0,current-best);
        corrMatrix=corrcoef(exogenousUnit,strategicUnit(:,1));
        rawCorrelation(i,dd)=corrMatrix(1,2);
        strategicUnitMean(i,dd)=mean(strategicUnit,'all');
        strategicUnitVariance(i,dd)=var(strategicUnit,1,'all');
    end
end
gapByKappa=max(playerGap,[],1);
correlationByKappa=mean(rawCorrelation,1);
meanByKappa=mean(strategicUnitMean,1);
varianceByKappa=mean(strategicUnitVariance,1);
end

function metrics=evaluate_true_metrics(x,p,kappa,nHeldout,seed)
stream=RandStream('mt19937ar','Seed',seed);
playerGap=zeros(p.N,1);congestion=zeros(nHeldout,1);
rawCorrelation=zeros(p.N,1);
strategicUnitMean=zeros(p.N,1);
strategicUnitVariance=zeros(p.N,1);
for i=1:p.N
    oi=[1:i-1,i+1:p.N];b=p.B(i,oi).';
    exogenousUnit=rand(stream,nHeldout,1);
    independentStrategicUnit=rand(stream,nHeldout,p.N-1);
    mixtureSelector=rand(stream,nHeldout,p.N-1)<kappa;
    strategicUnit=independentStrategicUnit;
    exogenousMatrix=repmat(exogenousUnit,1,p.N-1);
    strategicUnit(mixtureSelector)=exogenousMatrix(mixtureSelector);
    w=p.wL(i)+(p.wU(i)-p.wL(i))*exogenousUnit;
    dev=p.sigmaY*(2*strategicUnit-1);
    Y=min(1,max(0,x(oi).'+dev));
    wy=mean(w.*Y,1).';
    coeff=-p.r(i)*mean(w)+p.c(i)*b.'*wy;
    br=min(1,max(0,-coeff/p.a(i)));
    current=0.5*p.a(i)*x(i)^2+coeff*x(i);
    best=0.5*p.a(i)*br^2+coeff*br;
    playerGap(i)=max(0,current-best);
    congestion=congestion+p.c(i)*w*x(i).*(Y*b);
    corrMatrix=corrcoef(exogenousUnit,strategicUnit(:,1));
    rawCorrelation(i)=corrMatrix(1,2);
    strategicUnitMean(i)=mean(strategicUnit,'all');
    strategicUnitVariance(i)=var(strategicUnit,1,'all');
end
metrics.sneGap=max(playerGap);metrics.playerGaps=playerGap;
metrics.totalService=sum(x);metrics.congestionQ95=quantile(congestion,0.95);
metrics.meanCongestion=mean(congestion);
metrics.exogenousStrategicCorrelation=mean(rawCorrelation);
metrics.strategicUnitMean=mean(strategicUnitMean);
metrics.strategicUnitVariance=mean(strategicUnitVariance);
metrics.dependenceConstruction= ...
    'V=U with probability kappa; otherwise V is an independent Uniform(0,1)';
end

function ci=wilson_interval(successes,n,z)
phat=successes/n;den=1+z^2/n;center=(phat+z^2/(2*n))/den;
half=z/den*sqrt(phat*(1-phat)/n+z^2/(4*n^2));ci=[center-half,center+half];
end

function [slope,intercept,r2]=fit_loglog_scaling(x,y)
x=x(:); y=y(:);
if any(x<=0) || any(y<=0)
    error('DRGame:LogLogFitDomain','Log-log scaling data must be positive.');
end
coef=polyfit(log(x),log(y),1);
slope=coef(1); intercept=coef(2);
observed=log(y); predicted=polyval(coef,log(x));
r2=1-sum((observed-predicted).^2) ...
    /max(sum((observed-mean(observed)).^2),realmin);
end

function fig=publication_figure(withColorbar)
if withColorbar
    height=6.7;
else
    height=6.4;
end
fig=figure('Color','w','Units','centimeters', ...
    'Position',[2 2 8.9 height],'PaperPositionMode','auto');
end

function style_axes(ax)
set(ax,'FontName','Times New Roman','FontSize',8.1,'LineWidth',0.80, ...
    'Box','on','Color','w','XMinorTick','on','YMinorTick','on');
grid(ax,'on'); ax.GridAlpha=0.12; ax.MinorGridAlpha=0.06;
lgd=findall(ancestor(ax,'figure'),'Type','legend');
set(lgd,'FontName','Times New Roman','FontSize',7.3,'Box','off');
end

function files=export_standalone_figure(fig,outDir,stem,resolutionDpi)
files = struct('fig',[stem '.fig'],'png',[stem '.png'],'eps',[stem '.eps']);
drawnow;
savefig(fig,fullfile(outDir,files.fig));
exportgraphics(fig,fullfile(outDir,files.png), ...
    'Resolution',resolutionDpi,'BackgroundColor','white');
print(fig,fullfile(outDir,files.eps),'-depsc','-vector');
end

function draw_discrete_heatmap(ax,xCenters,yCenters,values)
% Render constant-color cells at the computed grid points.
xEdges=centered_edges(xCenters);
yEdges=centered_edges(yCenters);
for rr=1:numel(yCenters)
    for ee=1:numel(xCenters)
        patch(ax,[xEdges(ee) xEdges(ee+1) xEdges(ee+1) xEdges(ee)], ...
            [yEdges(rr) yEdges(rr) yEdges(rr+1) yEdges(rr+1)], ...
            values(rr,ee),'FaceColor','flat','EdgeColor',[0.92 0.92 0.92], ...
            'LineWidth',0.25,'HandleVisibility','off');
    end
end
end

function edges=centered_edges(centers)
centers=centers(:).';
midpoints=0.5*(centers(1:end-1)+centers(2:end));
edges=[centers(1)-0.5*(centers(2)-centers(1)),midpoints, ...
    centers(end)+0.5*(centers(end)-centers(end-1))];
end
