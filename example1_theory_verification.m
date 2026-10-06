%% Example 1: four-player edge-service game
% Distributed fixed-dual proxy dynamics and an offline DRNE benchmark.

clear; clc; close all;
rng(260826, 'twister');

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
outDir = fullfile(scriptDir, 'results', 'example1');
if ~isfolder(outDir)
    mkdir(outDir);
end

%% Game parameters and private samples
p.N = 4;
p.K = 5 * ones(p.N, 1);
p.a = [4.20; 4.60; 3.90; 4.80];
p.r = [2.45; 2.75; 2.20; 2.65];
p.c = [0.26; 0.31; 0.22; 0.29];
p.wL = [0.72; 0.68; 0.75; 0.70];
p.wU = [1.28; 1.34; 1.25; 1.31];
p.B = [0    0.45 0    0.30; ...
       0.38 0    0.32 0;    ...
       0    0.42 0    0.36; ...
       0.28 0    0.44 0];
p.B0 = p.B;

p.Acomm = [0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0];
p.L = diag(sum(p.Acomm, 2)) - p.Acomm;
lapEig = sort(real(eig(p.L)));
p.lambda2 = lapEig(2);
p.lambdaN = lapEig(end);

% Projection bounds the strategic second moment by sigmaY^2*(N-1)/3.
p.sigmaY = 0.16;
p.rhoCover = p.sigmaY * sqrt((p.N - 1) / 3);
p.rho = 1.05 * p.rhoCover * ones(p.N, 1);

% Set instance-specific radii from the known uniform law.
p.samples = cell(p.N, 1);
p.epsilon = zeros(p.N, 1);
p.actualW2 = zeros(p.N, 1);
for i = 1:p.N
    p.samples{i} = p.wL(i) + (p.wU(i) - p.wL(i)) * rand(p.K(i), 1);
    p.actualW2(i) = empirical_uniform_w2(p.samples{i}, p.wL(i), p.wU(i));
    p.epsilon(i) = 1.10 * p.actualW2(i) + 1e-4;
end

% Fixed multipliers regularize the samplewise inner problems.
p.lambdaXi = [10.0; 10.5; 9.5; 10.2];
p.lambdaY  = [10.0; 10.5; 9.5; 10.2];

% Reduce coupling if the analytical monotonicity margin fails.
couplingScale = 1;
for tuneAttempt = 1:20
    p.B = couplingScale * p.B0;
    [cst, cert] = analytical_constants(p);
    if cert.muF > 0.50
        break;
    end
    couplingScale = 0.90 * couplingScale;
end
if cert.muF <= 0.50
    error('DRGame:ParameterSearchFailed', ...
        'No parameter set with a positive analytical mu_F margin was found.');
end
p.couplingScale = couplingScale;

% Retain a strict 0.50 margin in the inner strong-concavity condition.
p.muZi = 2 * min(p.lambdaXi, p.lambdaY) - cst.ellZ - 0.50;
[cst, cert] = analytical_constants(p);
[gain, cst] = select_theorem_gains(p, cst, cert);
audit = build_theory_audit(p, cst, cert, gain);
assert_all_pass(audit);
%% Numerical monotonicity check
[cert.numericMuObserved, cert.numericAuditPoints] = ...
    numerical_monotonicity_audit(p, 1200);
if cert.numericMuObserved <= 0
    error('DRGame:NumericalMonotonicityAudit', ...
        'The sampled symmetric-Jacobian audit found a nonpositive eigenvalue.');
end

%% Offline proxy and robust equilibria
x0 = min(0.90, max(0.10, p.r .* cellfun(@mean, p.samples) ./ p.a));
[xProxy, proxySolve] = solve_proxy_ne(p, x0, 2e-11, 20000);
[proxyGap, proxyPlayerGaps] = proxy_nash_gap(xProxy, p);
if proxyGap > 2e-9
    error('DRGame:ProxySolve', 'Proxy Nash residual %.3e is too large.', proxyGap);
end

fprintf('\nComputing the offline exact DRNE by semidual best responses...\n');
[xDR, exactSolve] = solve_exact_drne(p, xProxy, 2e-10, 120);
[drGapExact, drPlayerGapsExact] = original_dr_nash_gap(xDR, p);
if drGapExact > 1e-8
    error('DRGame:ExactDRNESolve', ...
        'Offline exact DRNE residual %.3e exceeds 1e-8.', drGapExact);
end
[drGapProxy, drPlayerGapsProxy] = original_dr_nash_gap(xProxy, p);
proxyDrDistance = norm(xProxy - xDR);

%% Distributed two-time-scale dynamics
[X0, z0] = random_feasible_initial_state(p, 101);
Xeq = repmat(xProxy.', p.N, 1);
zEq = fixed_inner_map(Xeq, p);
y0 = [X0(:); z0];

tFinal = 16 / (gain.alpha * cert.muF);
tOut = unique([0; logspace(log10(max(gain.tau / 100, 1e-10)), ...
    log10(tFinal), 540).']);
odeOpts = odeset('RelTol', 2e-7, 'AbsTol', 2e-9, ...
    'MaxStep', tFinal / 350, 'NonNegative', []);

fprintf('Integrating the %d-state stiff ODE with ode15s (t_f=%.3e)...\n', ...
    numel(y0), tFinal);
tic;
[t, y] = ode15s(@(~, yy) distributed_rhs(yy, p, gain), ...
    tOut, y0, odeOpts);
mainOdeRuntime = toc;

traj = trajectory_metrics(t, y, xProxy, zEq, p);

%% Original-game Nash gap along the trajectory
gapIndex = unique(round(linspace(1, numel(t), 46)));
gapTime = t(gapIndex);
gapTrajectory = zeros(numel(gapIndex), 1);
fprintf('Evaluating the original-game DR Nash gap at %d raw time points...\n', ...
    numel(gapIndex));
for q = 1:numel(gapIndex)
    gapTrajectory(q) = original_dr_nash_gap(traj.xActual(gapIndex(q), :).', p);
end

%% Convergence from 100 feasible initial conditions
nInitial = 100;
initialConditionSeeds = (1001:1000+nInitial).';
multiRelative = zeros(numel(tOut), nInitial);
multiTerminal = zeros(nInitial, 1);
multiTerminalStrategyError = zeros(nInitial, 1);
fprintf('Running %d additional feasible initial conditions...\n', nInitial);
for run = 1:nInitial
    [Xr0, zr0] = random_feasible_initial_state(p, initialConditionSeeds(run));
    yr0 = [Xr0(:); zr0];
    [~, yr] = ode15s(@(~, yy) distributed_rhs(yy, p, gain), ...
        tOut, yr0, odeOpts);
    Er = composite_error_only(yr, xProxy, zEq, p);
    multiRelative(:, run) = Er / max(Er(1), realmin);
    multiTerminal(run) = multiRelative(end, run);
    Xterminal = reshape(yr(end,1:p.N^2),p.N,p.N);
    multiTerminalStrategyError(run) = norm(diag(Xterminal)-xProxy);
end
multiMedian = median(multiRelative, 2);
multiIQR = quantile(multiRelative,[0.25 0.75],2);
terminalSummary = struct( ...
    'minimumNormalizedError',min(multiTerminal), ...
    'medianNormalizedError',median(multiTerminal), ...
    'maximumNormalizedError',max(multiTerminal), ...
    'minimumStrategyError',min(multiTerminalStrategyError), ...
    'medianStrategyError',median(multiTerminalStrategyError), ...
    'maximumStrategyError',max(multiTerminalStrategyError));
if terminalSummary.maximumNormalizedError>2e-2 || ...
        terminalSummary.maximumStrategyError>1e-2
    error('DRGame:GlobalInitialConditionAudit', ...
        ['At least one deterministic initial condition did not reach the ', ...
         'common proxy-equilibrium neighborhood.']);
end

s = gain.alpha * t;
sOut = gain.alpha * tOut;
sGap = gain.alpha * gapTime;
[empiricalSlowRate, empiricalIntercept, fitR2, fitMask, fitInfo] = ...
    fit_exponential_rate(sOut, multiMedian);
theoreticalSlowRate = cst.c2 / gain.alpha;
empiricalTheoreticalRateRatio = empiricalSlowRate / theoreticalSlowRate;
if fitR2 < 0.995
    warning('DRGame:ExponentialFitR2', ...
        ['The log-linear fit R^2 is %.6f over s in [%.4g, %.4g]. ', ...
         'Inspect the fitting interval without altering the data.'], ...
        fitR2,fitInfo.interval(1),fitInfo.interval(2));
end
empiricalFit = exp(empiricalIntercept - empiricalSlowRate * sOut);
theoreticalRateReference = exp(-theoreticalSlowRate * sOut);
rigorousEnvelope = cst.c1 * exp(-cst.c2 * tOut);

%% Finite-time approximation bound
% Bound dual mismatch by ellLambda*(norm(lambdaBar)+Mlambda).
approxCert = certified_approximation_bound(p);
E0 = norm(X0-Xeq, 'fro') + norm(z0-zEq);
finiteCertificateByPlayer = approxCert.barOmegaPlayer + ...
    (2 * approxCert.ellPhi * cst.c1 * E0) .* exp(-cst.c2 * gapTime.');
omegaCertificateTrajectory = max(finiteCertificateByPlayer, [], 1).';
certificateGapRatio = approxCert.barOmega / max(drGapProxy, realmin);
certificateInformative = max(omegaCertificateTrajectory) <= ...
    100 * max([gapTrajectory; drGapProxy; 1e-12]);

%% Figures
colors = [0.00 0.35 0.62; 0.78 0.20 0.18; ...
          0.18 0.55 0.34; 0.48 0.30 0.64];
faint = [0.82 0.87 0.91];
figureResolutionDpi = 600;
figureFiles = struct;

fig1a = publication_figure();
ax = axes(fig1a); hold(ax, 'on');
for i = 1:p.N
    plot(ax, s, traj.xActual(:, i), '-', ...
        'Color', colors(i, :), 'LineWidth', 1.45);
    plot(ax, [s(1) s(end)], xProxy(i) * [1 1], '--', ...
        'Color', colors(i, :), 'LineWidth', 0.95, 'HandleVisibility', 'off');
    plot(ax, [s(1) s(end)], xDR(i) * [1 1], ':', ...
        'Color', colors(i, :), 'LineWidth', 1.20, 'HandleVisibility', 'off');
end
hTrajectoryKey = plot(ax, nan, nan, '-', 'Color', colors(1, :), ...
    'LineWidth', 1.45);
hProxyRef = plot(ax, nan, nan, 'k--', 'LineWidth', 1.0);
hDrRef = plot(ax, nan, nan, 'k:', 'LineWidth', 1.25);
xlabel(ax, 'Slow time $s=\alpha t$', 'Interpreter', 'latex');
ylabel(ax, 'Reserved service rate', 'Interpreter', 'latex');
xlim(ax, [0 max(s)]);
legend(ax, [hTrajectoryKey hProxyRef hDrRef], ...
    {'trajectories','proxy NE','exact DRNE'}, ...
    'Location', 'best');
style_axes(ax);
figureFiles.fig1a = export_standalone_figure( ...
    fig1a,outDir,'example1_fig1a',figureResolutionDpi);
close(fig1a);

fig1b = publication_figure();
ax = axes(fig1b); hold(ax, 'on');
semilogy(ax, s, max(traj.strategyError, realmin), '-', ...
    'Color', colors(1, :), 'LineWidth', 1.45);
semilogy(ax, s, max(traj.consensusError, realmin), '--', ...
    'Color', colors(2, :), 'LineWidth', 1.35);
semilogy(ax, s, max(traj.innerTrackingError, realmin), '-.', ...
    'Color', colors(3, :), 'LineWidth', 1.35);
xlabel(ax, 'Slow time $s=\alpha t$', 'Interpreter', 'latex');
ylabel(ax, 'Error', 'Interpreter', 'latex');
legend(ax, {'strategy-to-proxy','estimate disagreement','inner tracking'}, ...
    'Location', 'best');
set(ax, 'YScale', 'log');
style_axes(ax);
figureFiles.fig1b = export_standalone_figure( ...
    fig1b,outDir,'example1_fig1b',figureResolutionDpi);
close(fig1b);

fig1c = publication_figure();
ax = axes(fig1c); hold(ax, 'on');
for run = 1:nInitial
    semilogy(ax, sOut, max(multiRelative(:, run), realmin), '-', ...
        'Color', faint, 'LineWidth', 0.35, 'HandleVisibility', 'off');
end
hIQR = fill(ax,[sOut;flipud(sOut)], ...
    [max(multiIQR(:,1),realmin);flipud(max(multiIQR(:,2),realmin))], ...
    colors(1,:),'FaceAlpha',0.18,'EdgeColor','none');
hMedian = semilogy(ax, sOut, max(multiMedian, realmin), '-', ...
    'Color', colors(1, :), 'LineWidth', 1.65);
hFit = semilogy(ax, sOut(fitMask), max(empiricalFit(fitMask), realmin), ':', ...
    'Color', colors(2, :), 'LineWidth', 1.45);
hRate = semilogy(ax, sOut, max(theoreticalRateReference, realmin), ...
    'k--', 'LineWidth', 1.25);
xlabel(ax, 'Slow time $s=\alpha t$', 'Interpreter', 'latex');
ylabel(ax, '$E_r(s)/E_r(0)$', 'Interpreter', 'latex');
legend(ax, [hMedian hIQR hFit hRate], ...
    {'median','IQR','exponential fit','theoretical rate'}, ...
    'Location', 'best', 'Interpreter', 'latex');
set(ax, 'YScale', 'log');
style_axes(ax);
figureFiles.fig1c = export_standalone_figure( ...
    fig1c,outDir,'example1_fig1c',figureResolutionDpi);
close(fig1c);

fig1d = publication_figure();
ax = axes(fig1d); hold(ax, 'on');
gapPlotFloor = 1e-12;
hGap = semilogy(ax, sGap, max(gapTrajectory, gapPlotFloor), '-', ...
    'Color', colors(4, :), 'LineWidth', 1.50);
hFloor = semilogy(ax, [sGap(1) sGap(end)], max(drGapProxy, gapPlotFloor) * [1 1], ...
    'k--', 'LineWidth', 1.20);
xlabel(ax, 'Slow time $s=\alpha t$', 'Interpreter', 'latex');
ylabel(ax, 'Original DR Nash gap', 'Interpreter', 'latex');
legend(ax, [hGap hFloor], {'trajectory','proxy-equilibrium gap'}, ...
    'Location', 'best');
set(ax, 'YScale', 'log');
style_axes(ax);
figureFiles.fig1d = export_standalone_figure( ...
    fig1d,outDir,'example1_fig1d',figureResolutionDpi);
close(fig1d);

%% Save results
publicationAuditSummary = struct( ...
    'certifiedMuF',cert.muF, ...
    'observedMinimumSymmetricJacobianEigenvalue',cert.numericMuObserved, ...
    'lambda2Laplacian',p.lambda2, ...
    'gammaUsed',gain.gamma, ...
    'gammaLowerThreshold',gain.gammaMin, ...
    'alphaUsed',gain.alpha, ...
    'alphaMaximum',gain.alphaMax, ...
    'betaUsed',gain.beta, ...
    'betaMaximum',gain.betaMax, ...
    'tauUsed',gain.tau, ...
    'tauMaximum',gain.tauBar, ...
    'qx',gain.qx, ...
    'qz',gain.qz, ...
    'c1',cst.c1, ...
    'c2',cst.c2, ...
    'theoreticalSlowTimeRate',theoreticalSlowRate, ...
    'empiricalSlowTimeRate',empiricalSlowRate, ...
    'empiricalTheoreticalRateRatio',empiricalTheoreticalRateRatio, ...
    'fitR2',fitR2);
results = struct;
results.parameters = p;
results.constants = cst;
results.certificate = cert;
results.gains = gain;
results.theoryAudit = audit;
results.publicationAuditSummary = publicationAuditSummary;
results.xExactDRNE = xDR;
results.xProxyNE = xProxy;
results.proxySolve = proxySolve;
results.exactSolve = exactSolve;
results.proxyPlayerGaps = proxyPlayerGaps;
results.exactDRPlayerGaps = drPlayerGapsExact;
results.proxyOriginalDRPlayerGaps = drPlayerGapsProxy;
results.t = t;
results.slowTime = s;
results.trajectory = traj;
results.originalGapTime = gapTime;
results.originalGapSlowTime = sGap;
results.originalGapTrajectory = gapTrajectory;
results.initialConditionSeeds = initialConditionSeeds;
results.multiInitialRelativeErrors = multiRelative;
results.multiInitialMedian = multiMedian;
results.multiInitialIQR = multiIQR;
results.multiInitialTerminal = multiTerminal;
results.multiInitialTerminalStrategyError = multiTerminalStrategyError;
results.multiInitialTerminalSummary = terminalSummary;
results.numberOfInitialConditions = nInitial;
results.empiricalSlowTimeRate = empiricalSlowRate;
results.theoreticalSlowTimeRate = theoreticalSlowRate;
results.empiricalTheoreticalRateRatio = empiricalTheoreticalRateRatio;
results.empiricalFitR2 = fitR2;
results.empiricalFitMask = fitMask;
results.empiricalFitInfo = fitInfo;
results.rigorousEnvelope = rigorousEnvelope;
results.theoreticalRateReference = theoreticalRateReference;
results.theoreticalRateReferenceMeaning = ...
    'unit-prefactor slope reference exp(-(c2/alpha)s), not the rigorous pointwise envelope';
results.rigorousEnvelopeMeaning = 'c1*exp(-c2*t) theorem envelope';
results.approximationCertificate = approxCert;
results.initialCompositeErrorE0 = E0;
results.finiteTimeCertificateByPlayer = finiteCertificateByPlayer;
results.finiteTimeCertificate = omegaCertificateTrajectory;
results.certificateToActualGapRatio = certificateGapRatio;
results.certificateWouldBeInformative = certificateInformative;
results.certificatePlotted = false;
results.mainOdeRuntimeSeconds = mainOdeRuntime;
results.figureFiles = figureFiles;
results.figureResolutionDpi = figureResolutionDpi;
save(fullfile(outDir, 'example1_results.mat'), 'results', '-v7.3');

fprintf('\nExample 1 completed.\n');
fprintf('Proxy NE: %s\n', mat2str(xProxy.', 8));
fprintf('Reference DRNE: %s\n', mat2str(xDR.', 8));
fprintf('Proxy / reference Nash gaps: %.3e / %.3e\n', proxyGap, drGapExact);
fprintf('Original-game gap at proxy NE: %.6e\n', drGapProxy);
fprintf('Analytical / observed monotonicity bounds: %.6f / %.6f\n', ...
    cert.muF, cert.numericMuObserved);
fprintf('Terminal normalized error over %d initializations: %.3e to %.3e\n', ...
    nInitial, min(multiTerminal), max(multiTerminal));
fprintf('Results saved to: %s\n', outDir);

%% Local functions
function [cst, cert] = analytical_constants(p)
N = p.N;
cst.ellZ = zeros(N, 1);
cst.ellJ = zeros(N, 1);
cert.crossLipschitz = zeros(N);
cert.FcomponentLipschitz = zeros(N);
cert.muRows = zeros(N, 1);

for i = 1:N
    b = p.B(i, :).';
    nb = norm(b);
    bsum = sum(b);
    cst.ellZ(i) = p.c(i) * nb;
    mixedXW = max(p.r(i), abs(p.c(i) * bsum - p.r(i)));
    cst.ellJ(i) = sqrt(p.a(i)^2 + 2 * mixedXW^2 + ...
        2 * p.c(i)^2 * (p.wU(i)^2 + 1) * nb^2);

    if isfield(p, 'muZi')
        muInner = p.muZi(i);
    else
        muInner = max(2 * min(p.lambdaXi(i), p.lambdaY(i)) ...
            - cst.ellZ(i) - 0.50, 1e-3);
    end
    mZeta = 2 * p.lambdaXi(i) ...
        - p.c(i)^2 * nb^2 / (2 * p.lambdaY(i));
    if mZeta <= 0 || muInner <= 0
        cert.muF = -inf;
        return;
    end
    partialFz = mixedXW + p.wU(i) * p.c(i)^2 * nb^2 ...
        / (2 * p.lambdaY(i));
    gradFz = sqrt(mixedXW^2 + (p.c(i) * p.wU(i) * nb)^2);
    cert.FcomponentLipschitz(i, i) = p.a(i) + gradFz^2 / muInner;
    for j = 1:N
        if j ~= i && p.B(i, j) ~= 0
            Lij = p.c(i) * p.B(i, j) * ...
                (p.wU(i) + partialFz / mZeta);
            cert.crossLipschitz(i, j) = Lij;
            cert.FcomponentLipschitz(i, j) = Lij;
        end
    end
end

for i = 1:N
    pairPenalty = 0;
    for j = 1:N
        if j ~= i
            pairPenalty = pairPenalty + 0.5 * ...
                (cert.crossLipschitz(i, j) + cert.crossLipschitz(j, i));
        end
    end
    cert.muRows(i) = p.a(i) - pairPenalty;
end
cert.muF = min(cert.muRows);
cert.ellF = norm(cert.FcomponentLipschitz, 2);

if isfield(p, 'muZi')
    cst.muZ = min(p.muZi);
    cst.ellXzI = cst.ellJ + 2 * p.lambdaY;
    cst.ellGx = max(cst.ellJ);
    cst.ellGz = max(cst.ellJ ./ sqrt(p.K));
    cst.ellHx = max(sqrt(p.K) .* cst.ellXzI);
    cst.ellHz = max(cst.ellJ + 2 * max(p.lambdaXi, p.lambdaY));
    cst.ellH = cst.ellHx / cst.muZ;
    cst.ellFtilde = cst.ellGx + cst.ellGz * cst.ellH;
    cst.ellF = cert.ellF;
end
end

function [gain, cst] = select_theorem_gains(p, cst, cert)
gain.gammaMin = (cst.ellFtilde + ...
    (cst.ellF + cst.ellFtilde)^2 / (4 * cert.muF)) / p.lambda2;

% Maximize muGamma/ellGamma over theorem-admissible gamma.
factors = linspace(1.001, 8, 4000);
bestScore = -inf;
for k = 1:numel(factors)
    gamma = factors(k) * gain.gammaMin;
    Q = [cert.muF / p.N, -(cst.ellF + cst.ellFtilde) / (2 * sqrt(p.N)); ...
        -(cst.ellF + cst.ellFtilde) / (2 * sqrt(p.N)), ...
        gamma * p.lambda2 - cst.ellFtilde];
    muGamma = min(eig((Q + Q.') / 2));
    ellGamma = cst.ellFtilde + gamma * p.lambdaN;
    score = muGamma / ellGamma;
    if muGamma > 0 && score > bestScore
        bestScore = score;
        gain.gammaFactor = factors(k);
        gain.gamma = gamma;
        gain.Qgamma = Q;
        cst.muGamma = muGamma;
        cst.ellGamma = ellGamma;
    end
end
if ~isfinite(bestScore)
    error('DRGame:GammaSelection', 'No theorem-admissible gamma was found.');
end

gain.alphaMax = 2 * cst.muGamma / cst.ellGamma^2;
gain.betaMax = 2 * cst.muZ / cst.ellHz^2;
gain.alpha = cst.muGamma / cst.ellGamma^2;
gain.beta = cst.muZ / cst.ellHz^2;
gain.qx = sqrt(max(0, 1 - 2 * gain.alpha * cst.muGamma ...
    + gain.alpha^2 * cst.ellGamma^2));
gain.qz = sqrt(max(0, 1 - 2 * gain.beta * cst.muZ ...
    + gain.beta^2 * cst.ellHz^2));
gain.oneMinusQx = (2 * gain.alpha * cst.muGamma ...
    - gain.alpha^2 * cst.ellGamma^2) / (1 + gain.qx);
gain.oneMinusQz = (2 * gain.beta * cst.muZ ...
    - gain.beta^2 * cst.ellHz^2) / (1 + gain.qz);
gain.tauBar = gain.oneMinusQx * gain.oneMinusQz / ...
    (2 * gain.alpha * cst.ellH * cst.ellGz);
gain.tau = 0.35 * gain.tauBar;
gain.kappaLyapunov = gain.alpha * cst.ellGz / ...
    (cst.ellH * (1 + gain.qx));
gain.ThetaTau = [gain.oneMinusQx, -gain.alpha * cst.ellGz; ...
    -gain.alpha * cst.ellGz, gain.kappaLyapunov * ...
    (gain.oneMinusQz / gain.tau - gain.alpha * cst.ellH * cst.ellGz)];
cst.thetaMin = min(eig((gain.ThetaTau + gain.ThetaTau.') / 2));
cst.kappaMin = min(1, gain.kappaLyapunov);
cst.kappaMax = max(1, gain.kappaLyapunov);
cst.c1 = sqrt(cst.kappaMax / cst.kappaMin * ...
    ((1 + cst.ellH)^2 + 1) * (1 + cst.ellH^2));
cst.c2 = cst.thetaMin / cst.kappaMax;
end

function audit = build_theory_audit(p, cst, cert, gain)
name = strings(0, 1); value = zeros(0, 1); margin = zeros(0, 1); pass = false(0, 1);
    function add(n, v, m, tf)
        name(end+1,1) = string(n); value(end+1,1) = v; ...
            margin(end+1,1) = m; pass(end+1,1) = tf;
    end
add('compact strategy sets', 1, 1, true);
add('positive own-decision curvature', min(p.a), min(p.a), all(p.a > 0));
add('finite gradient Lipschitz bounds', max(cst.ellJ), min(cst.ellJ), ...
    all(isfinite(cst.ellJ) & cst.ellJ > 0));
for i = 1:p.N
    scMargin = 2 * min(p.lambdaXi(i), p.lambdaY(i)) - cst.ellZ(i) - p.muZi(i);
    add(sprintf('inner strong concavity player %d', i), p.muZi(i), ...
        scMargin, p.muZi(i) > 0 && scMargin > 0);
end
add('proxy strong monotonicity analytical certificate', cert.muF, cert.muF, cert.muF > 0);
add('communication graph connected', p.lambda2, p.lambda2, p.lambda2 > 0);
add('gamma theorem lower bound', gain.gamma, gain.gamma-gain.gammaMin, ...
    gain.gamma > gain.gammaMin);
add('Q_gamma positive definite', cst.muGamma, cst.muGamma, cst.muGamma > 0);
add('alpha theorem interval', gain.alpha, min(gain.alpha, gain.alphaMax-gain.alpha), ...
    gain.alpha > 0 && gain.alpha < gain.alphaMax);
add('beta theorem interval', gain.beta, min(gain.beta, gain.betaMax-gain.beta), ...
    gain.beta > 0 && gain.beta < gain.betaMax);
add('q_x contraction', gain.qx, 1-gain.qx, gain.qx >= 0 && gain.qx < 1);
add('q_z contraction', gain.qz, 1-gain.qz, gain.qz >= 0 && gain.qz < 1);
add('tau theorem interval', gain.tau, min(gain.tau, gain.tauBar-gain.tau), ...
    gain.tau > 0 && gain.tau < gain.tauBar);
add('Theta_tau positive definite', cst.thetaMin, cst.thetaMin, cst.thetaMin > 0);
rhoMargin = min(p.rho - p.rhoCover);
add('strategic coverage condition', min(p.rho), rhoMargin, rhoMargin >= -1e-14);
positiveRadiusMargin = min([p.epsilon;p.rho]);
add('positive radii for attained fixed-dual semidual', positiveRadiusMargin, ...
    positiveRadiusMargin, positiveRadiusMargin > 0);
audit = table(name, value, margin, pass, ...
    'VariableNames', {'Condition','Value','StrictMargin','Pass'});
end

function assert_all_pass(audit)
failed = audit(~audit.Pass, :);
if ~isempty(failed)
    disp(failed);
    error('DRGame:TheoryAuditFailed', 'A mandatory theorem condition failed.');
end
end

function [minimumEigenvalue, nPoints] = numerical_monotonicity_audit(p, nRandom)
rng(8421, 'twister');
X = rand(nRandom, p.N);
X = [X; zeros(1,p.N); ones(1,p.N); 0.5*ones(1,p.N)];
minimumEigenvalue = inf;
h = 2e-6;
for q = 1:size(X, 1)
    x = X(q, :).';
    J = zeros(p.N);
    for j = 1:p.N
        xp = x; xm = x;
        xp(j) = min(1, x(j) + h);
        xm(j) = max(0, x(j) - h);
        denom = xp(j) - xm(j);
        J(:, j) = (proxy_pseudogradient(xp, p) ...
            - proxy_pseudogradient(xm, p)) / denom;
    end
    minimumEigenvalue = min(minimumEigenvalue, ...
        min(eig((J + J.') / 2)));
end
nPoints = size(X, 1);
end

function [x, info] = solve_proxy_ne(p, x0, tol, maxIter)
x = min(1, max(0, x0(:)));
step = 0.90 * p.N / (sum(p.a) + p.N); % conservative initial scaling
step = min(step, 0.95 * 2 * 1 / max(p.a));
for iter = 1:maxIter
    F = proxy_pseudogradient(x, p);
    xNew = min(1, max(0, x - step * F));
    if norm(xNew - x, inf) < tol
        x = xNew;
        break;
    end
    x = xNew;
end
info.iterations = iter;
info.projectedResidual = norm(x - min(1, max(0, x - proxy_pseudogradient(x,p))), inf);
if iter == maxIter || info.projectedResidual > 2e-9
    % Refine the offline proxy equilibrium by bounded least squares.
    opts = optimoptions('lsqnonlin', 'Display', 'off', ...
        'FunctionTolerance', 1e-14, 'StepTolerance', 1e-14, ...
        'OptimalityTolerance', 1e-13, 'MaxIterations', 300, ...
        'MaxFunctionEvaluations', 10000);
    [x, ~, residual] = lsqnonlin(@(xx) proxy_pseudogradient(xx,p), ...
        x, zeros(p.N,1), ones(p.N,1), opts);
    info.lsqResidual = norm(residual, inf);
end
end

function F = proxy_pseudogradient(x, p)
F = zeros(p.N, 1);
for i = 1:p.N
    oi = [1:i-1, i+1:p.N];
    [~, zeta, v] = inner_max_batch(x(i), x(oi), p.samples{i}, ...
        p.lambdaXi(i), p.lambdaY(i), i, p);
    congestion = v * p.B(i, oi).';
    F(i) = mean(p.a(i) * x(i) - p.r(i) * zeta ...
        + p.c(i) * zeta .* congestion);
end
end

function [gap, playerGaps] = proxy_nash_gap(x, p)
playerGaps = zeros(p.N, 1);
opt = optimset('Display', 'off', 'TolX', 2e-12, 'MaxIter', 500);
for i = 1:p.N
    current = proxy_cost_i(x(i), x, i, p);
    [~, best] = fminbnd(@(u) proxy_cost_i(u, x, i, p), 0, 1, opt);
    playerGaps(i) = max(0, current - best);
end
gap = max(playerGaps);
end

function value = proxy_cost_i(xi, x, i, p)
oi = [1:i-1, i+1:p.N];
vals = inner_max_batch(xi, x(oi), p.samples{i}, ...
    p.lambdaXi(i), p.lambdaY(i), i, p);
value = p.lambdaXi(i) * p.epsilon(i)^2 ...
    + p.lambdaY(i) * p.rho(i)^2 + mean(vals);
end

function [x, info] = solve_exact_drne(p, x0, tol, maxSweeps)
x = x0(:);
lambdaWarm = [p.lambdaXi, p.lambdaY];
relax = 0.88;
history = nan(maxSweeps, 2);
for sweep = 1:maxSweeps
    xOld = x;
    for i = 1:p.N
        [br, lambdaBr] = exact_player_best_response(i, x, p, lambdaWarm(i,:));
        x(i) = relax * br + (1-relax) * x(i);
        lambdaWarm(i,:) = lambdaBr;
    end
    history(sweep,1) = norm(x-xOld, inf);
    if mod(sweep, 4) == 0 || history(sweep,1) < tol
        history(sweep,2) = original_dr_nash_gap(x, p);
        if history(sweep,1) < tol && history(sweep,2) < 1e-8
            break;
        end
    end
end
info.sweeps = sweep;
info.history = history(1:sweep,:);
info.lambdaWarm = lambdaWarm;
end

function [gap, playerGaps] = original_dr_nash_gap(x, p)
playerGaps = zeros(p.N, 1);
for i = 1:p.N
    current = exact_robust_cost_i(x(i), x, i, p, ...
        [p.lambdaXi(i), p.lambdaY(i)]);
    [~, ~, best] = exact_player_best_response(i, x, p, ...
        [p.lambdaXi(i), p.lambdaY(i)]);
    playerGaps(i) = max(0, current - best);
end
gap = max(playerGaps);
end

function [bestX, bestLambda, bestValue] = exact_player_best_response(i, x, p, lambda0)
oi = [1:i-1, i+1:p.N];
u0 = [x(i); max(lambda0(:), 1e-3)];
ubLambda = 1000;
lb = [0; 0; 0]; ub = [1; ubLambda; ubLambda];
opts = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
    'SpecifyObjectiveGradient', true, 'OptimalityTolerance', 2e-11, ...
    'StepTolerance', 2e-12, 'ConstraintTolerance', 1e-12, ...
    'MaxIterations', 300, 'MaxFunctionEvaluations', 8000);
fun = @(u) semidual_player_objective(u, x(oi), p.samples{i}, ...
    p.epsilon(i), p.rho(i), i, p);
try
    [u, bestValue] = fmincon(fun, u0, [], [], [], [], lb, ub, [], opts);
catch
    opts.Algorithm = 'interior-point';
    [u, bestValue] = fmincon(fun, u0, [], [], [], [], lb, ub, [], opts);
end
if any(u(2:3) > 0.95 * ubLambda)
    error('DRGame:DualUpperBound', 'Exact semidual multiplier hit its numerical upper bound.');
end
bestX = u(1);
bestLambda = u(2:3).';
end

function value = exact_robust_cost_i(xi, x, i, p, lambda0)
oi = [1:i-1, i+1:p.N];
opts = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
    'SpecifyObjectiveGradient', true, 'OptimalityTolerance', 2e-11, ...
    'StepTolerance', 2e-12, 'ConstraintTolerance', 1e-12, ...
    'MaxIterations', 250, 'MaxFunctionEvaluations', 6000);
fun = @(lam) semidual_lambda_objective(lam, xi, x(oi), p.samples{i}, ...
    p.epsilon(i), p.rho(i), i, p);
[lam, value] = fmincon(fun, max(lambda0(:),1e-3), [], [], [], [], ...
    [0;0], [1000;1000], [], opts);
if any(lam > 950)
    error('DRGame:DualUpperBound', 'Exact cost multiplier hit its numerical upper bound.');
end
end

function [f, grad] = semidual_player_objective(u, xminus, samples, epsVal, rhoVal, i, p)
xi = u(1); lambda = u(2:3);
[f, gx, glambda] = semidual_eval(xi, xminus, samples, ...
    epsVal, rhoVal, lambda, i, p);
grad = [gx; glambda];
end

function [f, grad] = semidual_lambda_objective(lambda, xi, xminus, samples, epsVal, rhoVal, i, p)
[f, ~, grad] = semidual_eval(xi, xminus, samples, ...
    epsVal, rhoVal, lambda, i, p);
end

function [f, gx, glambda] = semidual_eval(xi, xminus, samples, epsVal, rhoVal, lambda, i, p)
[vals, zeta, v] = inner_max_batch(xi, xminus, samples, ...
    lambda(1), lambda(2), i, p);
b = p.B(i, [1:i-1, i+1:p.N]).';
f = lambda(1) * epsVal^2 + lambda(2) * rhoVal^2 + mean(vals);
gx = mean(p.a(i) * xi - p.r(i) * zeta + p.c(i) * zeta .* (v*b));
glambda = [epsVal^2 - mean((zeta-samples).^2); ...
    rhoVal^2 - mean(sum((v-xminus(:).').^2, 2))];
end

function [values, zetaBest, vBest] = inner_max_batch(xi, xminus, samples, lambdaXi, lambdaY, i, p)
% Enumerate saturation intervals and maximize each scalar quadratic.
xminus = xminus(:).'; samples = samples(:);
b = p.B(i, [1:i-1, i+1:p.N]);
K = numel(samples); d = numel(b);
wL = p.wL(i); wU = p.wU(i);

breaks = [wL; wU];
if lambdaY > 1e-13 && xi > 1e-13
    qcoef = xi * p.c(i) * b;
    for j = 1:d
        if qcoef(j) > 0
            zbreak = 2 * lambdaY * (1-xminus(j)) / qcoef(j);
            if zbreak > wL && zbreak < wU
                breaks(end+1,1) = zbreak; %#ok<AGROW>
            end
        end
    end
end
breaks = unique(sort(breaks));

best = -inf(K,1); zetaBest = wL * ones(K,1);
for seg = 1:numel(breaks)-1
    left = breaks(seg); right = breaks(seg+1);
    mid = 0.5 * (left+right);
    if right-left < 5e-14, continue; end
    if lambdaY > 1e-13
        vMid = min(1, max(0, xminus + xi*p.c(i)*mid*b/(2*lambdaY)));
        unsat = vMid < 1-1e-10 & vMid > 1e-10;
    else
        unsat = false(1,d);
    end
    A = -lambdaXi;
    Bbase = -p.r(i)*xi;
    for j = 1:d
        qj = xi*p.c(i)*b(j);
        if lambdaY <= 1e-13
            if qj >= 0, Bbase = Bbase + qj; end
        elseif unsat(j)
            A = A + qj^2/(4*lambdaY);
            Bbase = Bbase + qj*xminus(j);
        else
            Bbase = Bbase + qj;
        end
    end
    Bvec = Bbase + 2*lambdaXi*samples;
    candidates = [left*ones(K,1), right*ones(K,1)];
    if A < -1e-14
        vertex = -Bvec/(2*A);
        candidates(:,3) = min(right, max(left, vertex));
    end
    for cc = 1:size(candidates,2)
        zc = candidates(:,cc);
        vc = adversarial_v(xi, xminus, zc, lambdaY, b, p.c(i));
        val = 0.5*p.a(i)*xi^2 - p.r(i)*xi*zc ...
            + p.c(i)*xi*zc.*(vc*b.') - lambdaXi*(zc-samples).^2 ...
            - lambdaY*sum((vc-xminus).^2,2);
        take = val > best;
        best(take) = val(take);
        zetaBest(take) = zc(take);
    end
end
vBest = adversarial_v(xi, xminus, zetaBest, lambdaY, b, p.c(i));
values = best;
end

function v = adversarial_v(xi, xminus, zeta, lambdaY, b, ci)
K = numel(zeta);
if lambdaY > 1e-13
    v = repmat(xminus, K, 1) ...
        + (xi*ci/(2*lambdaY)) * (zeta(:)*b);
    v = min(1, max(0, v));
else
    v = repmat(xminus, K, 1);
    positive = xi*ci*(zeta(:)*b) >= 0;
    v(positive) = 1;
end
end

function [X0, z0] = random_feasible_initial_state(p, seed)
rng(seed, 'twister');
X0 = 0.05 + 0.90*rand(p.N, p.N);
z0 = zeros(sum(p.K)*p.N, 1);
cursor = 0;
for i = 1:p.N
    for k = 1:p.K(i)
        oi = [1:i-1, i+1:p.N]; %#ok<NASGU>
        z0(cursor+1) = p.wL(i)+(p.wU(i)-p.wL(i))*rand;
        z0(cursor+2:cursor+p.N) = rand(p.N-1,1);
        cursor = cursor+p.N;
    end
end
end

function z = fixed_inner_map(X, p)
z = zeros(sum(p.K)*p.N, 1);
cursor = 0;
for i = 1:p.N
    oi = [1:i-1, i+1:p.N];
    [~, zeta, v] = inner_max_batch(X(i,i), X(i,oi), p.samples{i}, ...
        p.lambdaXi(i), p.lambdaY(i), i, p);
    for k = 1:p.K(i)
        z(cursor+1:cursor+p.N) = [zeta(k); v(k,:).'];
        cursor = cursor+p.N;
    end
end
end

function dy = distributed_rhs(y, p, gain)
nX = p.N^2;
X = reshape(y(1:nX), p.N, p.N);
z = y(nX+1:end);
C = p.L*X;
Gmat = zeros(p.N);
dZ = zeros(size(z));
cursor = 0;
for i = 1:p.N
    oi = [1:i-1, i+1:p.N];
    zeta = zeros(p.K(i),1); V = zeros(p.K(i),p.N-1);
    for k = 1:p.K(i)
        block = z(cursor+1:cursor+p.N);
        zeta(k) = block(1); V(k,:) = block(2:end).';
        cursor = cursor+p.N;
    end
    congestion = V*p.B(i,oi).';
    Gmat(i,i) = mean(p.a(i)*X(i,i)-p.r(i)*zeta ...
        + p.c(i)*zeta.*congestion);
end
Xtarget = min(1, max(0, X-gain.alpha*(Gmat+gain.gamma*C)));
dX = Xtarget-X;

cursor = 0;
for i = 1:p.N
    oi = [1:i-1, i+1:p.N];
    b = p.B(i,oi).';
    for k = 1:p.K(i)
        block = z(cursor+1:cursor+p.N);
        zeta = block(1); v = block(2:end);
        H = [X(i,i)*(-p.r(i)+p.c(i)*(b.'*v)) ...
            - 2*p.lambdaXi(i)*(zeta-p.samples{i}(k)); ...
            X(i,i)*p.c(i)*zeta*b ...
            - 2*p.lambdaY(i)*(v-X(i,oi).')];
        target = block+gain.beta*H;
        target(1) = min(p.wU(i), max(p.wL(i), target(1)));
        target(2:end) = min(1, max(0, target(2:end)));
        dZ(cursor+1:cursor+p.N) = (target-block)/gain.tau;
        cursor = cursor+p.N;
    end
end
dy = [dX(:); dZ];
end

function traj = trajectory_metrics(t, y, xProxy, zEq, p)
nT = numel(t); nX = p.N^2;
traj.xActual = zeros(nT,p.N);
traj.strategyError = zeros(nT,1);
traj.consensusError = zeros(nT,1);
traj.innerTrackingError = zeros(nT,1);
traj.compositeError = zeros(nT,1);
Xeq = repmat(xProxy.',p.N,1);
Pperp = eye(p.N)-ones(p.N)/p.N;
for q = 1:nT
    X = reshape(y(q,1:nX),p.N,p.N);
    z = y(q,nX+1:end).';
    xa = diag(X);
    hX = fixed_inner_map(X,p);
    traj.xActual(q,:) = xa.';
    traj.strategyError(q) = norm(xa-xProxy);
    traj.consensusError(q) = norm(Pperp*X,'fro');
    traj.innerTrackingError(q) = norm(z-hX);
    traj.compositeError(q) = norm(X-Xeq,'fro')+norm(z-zEq);
end
traj.compositeRelative = traj.compositeError/max(traj.compositeError(1),realmin);
end

function E = composite_error_only(y, xProxy, zEq, p)
nT = size(y,1); nX = p.N^2; Xeq = repmat(xProxy.',p.N,1);
E = zeros(nT,1);
for q = 1:nT
    X = reshape(y(q,1:nX),p.N,p.N);
    z = y(q,nX+1:end).';
    E(q) = norm(X-Xeq,'fro')+norm(z-zEq);
end
end

function [rate, intercept, r2, mask, info] = fit_exponential_rate(t, relativeError)
% Fit after the fast layer and before the terminal numerical floor.
t = t(:); relativeError = relativeError(:);
minPoints = 60;
belowTransient = find(relativeError <= 0.65,1,'first');
if isempty(belowTransient), belowTransient = max(2,round(0.05*numel(t))); end
startTime = max(0.02*t(end),t(belowTransient));
tailStart = max(1,numel(relativeError)-4);
floorLevel = max(1e-12,2*median(relativeError(tailStart:end)));
mask = t >= startTime & t <= 0.90*t(end) & relativeError > floorLevel;
if nnz(mask) < minPoints
    mask = t >= 0.05*t(end) & t <= 0.85*t(end) ...
        & relativeError > max(1e-12,relativeError(end));
end
if nnz(mask) < minPoints
    error('DRGame:ExponentialFitWindow', ...
        'Only %d fit points remain; at least %d are required.',nnz(mask),minPoints);
end
coef = polyfit(t(mask),log(max(relativeError(mask),realmin)),1);
rate = max(0,-coef(1)); intercept = coef(2);
pred = polyval(coef,t(mask)); obs = log(max(relativeError(mask),realmin));
r2 = 1-sum((obs-pred).^2)/max(sum((obs-mean(obs)).^2),realmin);
info.interval = [min(t(mask)),max(t(mask))];
info.nPoints = nnz(mask);
info.floorLevel = floorLevel;
info.minimumRequiredPoints = minPoints;
info.selectionRule = ['after 2% slow-time/boundary layer and E<=0.65; ', ...
    'before 90% horizon and terminal floor'];
end

function cert = certified_approximation_bound(p)
% Bound dual mismatch using norm(lambdaStar) <= Mlambda.
N = p.N;
cert.ellLambda = zeros(N,1);
cert.Jlower = zeros(N,1);
cert.Jupper = zeros(N,1);
cert.Mlambda = zeros(N,1);
cert.rLambdaBound = zeros(N,1);
cert.barOmegaPlayer = zeros(N,1);
cert.ellPhi = zeros(N,1);

for i = 1:N
    oi = [1:i-1, i+1:N];
    b = p.B(i,oi);
    d = numel(oi);
    corners = dec2bin(0:2^d-1)-'0';

    % Convexity and multilinearity place the maximum at support vertices.
    jUpper = -inf;
    ownGradMax = 0;
    for xi = [0 1]
        for zeta = [p.wL(i) p.wU(i)]
            for row = 1:size(corners,1)
                v = corners(row,:);
                value = 0.5*p.a(i)*xi^2-p.r(i)*zeta*xi ...
                    +p.c(i)*zeta*xi*(b*v.');
                jUpper = max(jUpper,value);
                ownGrad = p.a(i)*xi-p.r(i)*zeta ...
                    +p.c(i)*zeta*(b*v.');
                ownGradMax = max(ownGradMax,abs(ownGrad));
            end
        end
    end

    % Minimize the empirical nominal cost over strategy vertices.
    meanW = mean(p.samples{i});
    jLower = inf;
    for row = 1:size(corners,1)
        xminus = corners(row,:);
        linear = meanW*(-p.r(i)+p.c(i)*(b*xminus.'));
        xiStar = min(1,max(0,-linear/p.a(i)));
        value = 0.5*p.a(i)*xiStar^2+linear*xiStar;
        jLower = min(jLower,value);
    end

    radiusFloor = min(p.epsilon(i)^2,p.rho(i)^2);
    if radiusFloor <= 0
        error('DRGame:CertificatePositiveRadius', ...
            'The explicit dual bound requires positive epsilon_i and rho_i.');
    end
    diameterXi = p.wU(i)-p.wL(i);
    diameterMinus = sqrt(d);
    ellLambda = hypot(p.epsilon(i)^2+diameterXi^2, ...
        p.rho(i)^2+diameterMinus^2);
    Mlambda = (jUpper-jLower)/radiusFloor;
    rLambdaBound = norm([p.lambdaXi(i),p.lambdaY(i)])+Mlambda;

    cert.ellLambda(i) = ellLambda;
    cert.Jlower(i) = jLower;
    cert.Jupper(i) = jUpper;
    cert.Mlambda(i) = Mlambda;
    cert.rLambdaBound(i) = rLambdaBound;
    cert.barOmegaPlayer(i) = ellLambda*rLambdaBound;

    % Bound opponent derivatives by 2*lambdaY using Danskin's formula.
    cert.ellPhi(i) = hypot(ownGradMax,2*p.lambdaY(i)*sqrt(d));
end
cert.barOmega = max(cert.barOmegaPlayer);
cert.ellPhiMax = max(cert.ellPhi);
cert.boundDescription = ['Proposition-14 upper bound using ', ...
    'r_lambda <= ||bar_lambda|| + M_lambda'];
end

function d = empirical_uniform_w2(samples, lo, hi)
s = sort(samples(:)); K = numel(s); width = hi-lo; total = 0;
for k = 1:K
    u0 = (k-1)/K; u1 = k/K; a = s(k)-lo;
    total = total + a^2*(u1-u0) - a*width*(u1^2-u0^2) ...
        + width^2/3*(u1^3-u0^3);
end
d = sqrt(max(total,0));
end

function fig = publication_figure()
fig = figure('Color','w','Units','centimeters', ...
    'Position',[2 2 8.9 6.5],'PaperPositionMode','auto');
end

function style_axes(ax)
set(ax,'FontName','Times New Roman','FontSize',8.2, ...
    'LineWidth',0.80,'Box','on','Color','w', ...
    'XMinorTick','on','YMinorTick','on');
grid(ax,'on');
ax.GridAlpha = 0.12;
ax.MinorGridAlpha = 0.06;
lgd = findall(ancestor(ax,'figure'),'Type','legend');
set(lgd,'FontName','Times New Roman','FontSize',7.5,'Box','off');
end

function files = export_standalone_figure(fig,outDir,stem,resolutionDpi)
files = struct('fig',[stem '.fig'],'png',[stem '.png'],'eps',[stem '.eps']);
drawnow;
savefig(fig,fullfile(outDir,files.fig));
exportgraphics(fig,fullfile(outDir,files.png), ...
    'Resolution',resolutionDpi,'BackgroundColor','white');
print(fig,fullfile(outDir,files.eps),'-depsc','-vector');
end
