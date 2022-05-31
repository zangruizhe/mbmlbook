module FSharpVersion.Common

open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.Models.Attributes
open Microsoft.ML.Probabilistic.Distributions

let engine = InferenceEngine()
engine.ShowFactorGraph <- true
engine.NumberOfIterations <- 10

let Infer<'T> (name: string) (engine: InferenceEngine) prior =
    let r = engine.Infer<'T>(prior)
    printfn $"Posterior {name}=%A{r}"
    r

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

let GetArrayOfArrayVar<'T> name (single_row_range: Range) (num_row: Range) =
    Variable
        .Array<'T>(Variable.Array<'T>(single_row_range), num_row)
        .Named(name)

let Get2DArrayVar<'T> name (row: Range) (column: Range) =
    Variable.Array<'T>(row, column).Named(name)

let Get3DArrayVar<'T> name (row: Range) (column: Range) (threeD: Range) =
    let x = Variable.Array<'T>(row, column)

    let y =
        Variable.Array<'T>(x, threeD).Named(name)

    y // :> VariableArray<VariableArray2D<'T>, 'T [] [,]>


let GetVarFromDist<'T, 'D when 'D :> IDistribution<'T>> (name: string) (prior: Variable<'D>) =
    Variable.Random<'T, 'D>(prior).Named(name)

let GetVarWithDist<'T, 'D when 'D :> IDistribution<'T>> (name: string) =
    let prior = GetVar<'D>(name + "Prior")
    prior, Variable.Random<'T, 'D>(prior).Named(name)


let GetArrayVarWithDst<'T, 'D when 'D :> IDistribution<'T>> (name: string) (range: Range) =
    //    let priors = Variable.New<DistributionStructArray<Gaussian, double>>().Named(name+"Priors")
    let priors =
        GetArrayVar<'D> (name + "Priors") range

    let value = GetArrayVar<'T> name range
    value.[range] <- Variable.Random(priors.[range])
    priors, value

let GetArrayOfArrayVarWithDst<'T, 'D when 'D :> IDistribution<'T>> (name: string) (column: Range) (row: Range) =
    //    let priors = Variable.New<DistributionStructArray<Gaussian, double>>().Named(name+"Priors")
    let priors =
        GetArrayOfArrayVar<'D> (name + "Priors") column row

    let value =
        GetArrayOfArrayVar<'T> name column row

    value.[row][column] <- Variable.Random(priors.[row][column])
    priors, value

let Get2DVarWithDst<'T, 'D when 'D :> IDistribution<'T>> (name: string) (row: Range) (column: Range) =
    let priors =
        Get2DArrayVar<'D> (name + "Priors") row column

    let value =
        Get2DArrayVar<'T> name row column

    value.[row, column] <- Variable.Random(priors.[row, column])
    priors, value

let Get3DVarWithDst<'T, 'D when 'D :> IDistribution<'T>> (name: string) (row: Range) (column: Range) (threeD: Range) =
    let priors =
        Get3DArrayVar<'D> (name + "Priors") row column threeD

    let value =
        Get3DArrayVar<'T> name row column threeD

    value.[threeD][row, column] <- Variable.Random(priors.[threeD][row, column])
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

let GetGaussianArray (name: string) (range: Range) =
    GetArrayVarWithDst<float, Gaussian> name range

//let GetGaussianArray (mean: Variable<float>) (precise: Variable<float>) (len: Variable<int>) =
//    let range = Range len
//
//    Variable.AssignVariableArray
//        (Variable.Array<float> range)
//        range
//        (fun _ -> Variable.GaussianFromMeanAndPrecision(mean, precise))
let ForeachBlockI (r: Range) (body: ForEachBlock -> unit) =
    let block = Variable.ForEach(r)
    body block
    block.Dispose()
