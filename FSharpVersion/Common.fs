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

let GetVar<'T> name = Variable.New<'T>().Named(name)
let GetGaussian name = GetVar<Gaussian> name

let GetVarFromDist<'T, 'D when 'D :> IDistribution<'T>> (name: string) (prior: Variable<'D>) =
    Variable.Random<'T, 'D>(prior).Named(name)
    
let GetVarWithDist<'T, 'D when 'D :> IDistribution<'T>> (name: string) =
    let prior = GetVar<'D> (name+"Prior")
    prior, Variable.Random<'T, 'D>(prior).Named(name)
    
let GetVarWithGaussian (name:string) =
    let prior = GetVar<Gaussian> (name+"Prior")
    let var = Variable.Random(prior).Named(name)
    prior, var
    

let GetGaussianArray (mean: Variable<float>) (precise: Variable<float>) (len: Variable<int>) =
    let range = Range len

    Variable.AssignVariableArray
        (Variable.Array<float> range)
        range
        (fun _ -> Variable.GaussianFromMeanAndPrecision(mean, precise))
