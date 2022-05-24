module FSharpVersion.Common

open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Models.Attributes
open Microsoft.ML.Probabilistic.Distributions

let engine = InferenceEngine()
engine.ShowFactorGraph <- true
engine.NumberOfIterations <- 10

let GetRangeLen name =
    Variable
        .New<int>()
        .Named(name)
        .Attrib(DoNotInfer())

let GetRange (name: string) =
    let len = GetRangeLen("num" + name)
    len, Range(len).Named(name)

let GetVarArray<'T> name (range: Range) = Variable.Array<'T>(range).Named(name)

let GetGaussianVar name = Variable.New<Gaussian>().Named(name)

let GetVarFromGaussian (name: string) (prior: Variable<Gaussian>) =
    Variable.Random(prior).Named(name)

let GetGamma (name: string) (prior: Variable<Gamma>) =
    let tmp = Variable<float>.Random<Gamma> prior
    tmp.Named(name)

let GetGaussianArray (mean: Variable<float>) (precise: Variable<float>) (len: Variable<int>) =
    let range = Range len

    Variable.AssignVariableArray(Variable.Array<float> range) range (fun d ->
        Variable.GaussianFromMeanAndPrecision(mean, precise))
