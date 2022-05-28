module FSharpVersion.Common

open Microsoft.ML.Probabilistic.Models
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

let GetVar<'T> name = Variable.New<'T>().Named(name)

let GetArrayVar<'T> name (range: Range) = Variable.Array<'T>(range).Named(name)

let GetArrayOfArrayVar<'T> name (column: Range) (row: Range) =
    Variable
        .Array<'T>(Variable.Array<'T>(column), row)
        .Named(name)
        
let Get3DArrayVar<'T> name (column: Range) (row: Range) (threeD: Range)=
    Variable.Array<'T>(Variable.Array<'T>(column, row), threeD).Named(name)

let GetVarFromDist<'T, 'D when 'D :> IDistribution<'T>> (name: string) (prior: Variable<'D>) =
    Variable.Random<'T, 'D>(prior).Named(name)

let GetVarWithDist<'T, 'D when 'D :> IDistribution<'T>> (name: string) =
    let prior = GetVar<'D>(name + "Prior")
    prior, Variable.Random<'T, 'D>(prior).Named(name)


let GetArrayDst<'T, 'D when 'D :> IDistribution<'T>> (name: string) (range: Range) =
    //    let priors = Variable.New<DistributionStructArray<Gaussian, double>>().Named(name+"Priors")
    let priors =
        Variable.Array<'D>(range).Named(name + "Priors")

    let value = Variable.Array<'T>(range).Named(name)

    value.[range] <- Variable.Random(priors.[range])

    priors, value

let GetGaussian name = GetVar<Gaussian> name

let GetVarWithGaussian (name: string) = GetVarWithDist<float, Gaussian>(name)

let AddDoNotInfer<'T> (value: Variable<'T>) = value.AddAttribute(DoNotInfer())

let GetGaussianArrayV0 (name: string) (range: Range) =
    let priors =
        Variable
            .New<DistributionStructArray<Gaussian, double>>()
            .Named(name + "Priors")

    let value =
        Variable.Array<double>(range).Named(name)

    value.SetTo(Variable.Random(priors))
    priors, value

let GetGaussianArray (name: string) (range: Range) = GetArrayDst<float, Gaussian> name range

//let GetGaussianArray (mean: Variable<float>) (precise: Variable<float>) (len: Variable<int>) =
//    let range = Range len
//
//    Variable.AssignVariableArray
//        (Variable.Array<float> range)
//        range
//        (fun _ -> Variable.GaussianFromMeanAndPrecision(mean, precise))
