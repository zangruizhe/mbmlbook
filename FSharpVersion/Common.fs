module FSharpVersion.Common

open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Models.Attributes

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
