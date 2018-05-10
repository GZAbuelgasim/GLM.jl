__precompile__()

module GLM
    using Distributions, Reexport
    using Base.LinAlg.LAPACK: potrf!, potrs!
    using Base.LinAlg.BLAS: gemm!, gemv!
    using Base.LinAlg: copytri!, QRCompactWY, Cholesky, CholeskyPivoted, BlasReal
    using StatsBase: StatsBase, CoefTable, StatisticalModel, RegressionModel
    using StatsFuns: logit, logistic
    @reexport using StatsModels
    using Distributions: sqrt2, sqrt2π
    using Compat

    import Base: (\), cholfact, convert, cor, show, size
    import StatsBase: coef, coeftable, confint, deviance, nulldeviance, dof, dof_residual,
                      loglikelihood, nullloglikelihood, nobs, stderror, vcov, residuals, predict,
                      fit, model_response, r2, r², adjr2, adjr², PValue
    import StatsFuns: xlogy
    import SpecialFunctions: erfc, erfcinv
    export coef, coeftable, confint, deviance, nulldeviance, dof, dof_residual,
           loglikelihood, nullloglikelihood, nobs, stderror, vcov, residuals, predict,
           fit, fit!, model_response, r2, r², adjr2, adjr²

    export
        # types
        ## Distributions
        Bernoulli,
        Binomial,
        Gamma,
        Gaussian,
        InverseGaussian,
        Normal,
        Poisson,

        ## Link types
        CauchitLink,
        CloglogLink,
        IdentityLink,
        InverseLink,
        InverseSquareLink,
        LogitLink,
        LogLink,
        ProbitLink,
        SqrtLink,

        # Model types
        GeneralizedLinearModel,
        LinearModel,

        # functions
        canonicallink,  # canonical link function for a distribution
        deviance,       # deviance of fitted and observed responses
        devresid,       # vector of squared deviance residuals
        formula,        # extract the formula from a model
        glm,            # general interface
        linpred,        # linear predictor
        lm,             # linear model
        nobs,           # total number of observations
        predict,        # make predictions
        ftest           # compare models with an F test

    const FP = AbstractFloat
    const FPVector{T<:FP} = AbstractArray{T,1}

    """
        ModResp

    Abstract type representing a model response vector
    """
    abstract type ModResp end                         # model response

    """
        LinPred

    Abstract type representing a linear predictor
    """
    abstract type LinPred end                         # linear predictor in statistical models
    abstract type DensePred <: LinPred end            # linear predictor with dense X
    abstract type LinPredModel <: RegressionModel end # model based on a linear predictor

    include("linpred.jl")
    include("lm.jl")
    include("glmtools.jl")
    include("glmfit.jl")
    include("ftest.jl")

end # module
