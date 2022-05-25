module FSharpVersion.Chapter4

open Microsoft.ML.Probabilistic.Distributions.Automata
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Models.Attributes

open FSharpVersion.Common
open MeetingYourMatch
open MeetingYourMatch.Items
open MBMLViews

[<Literal>]
let DataPath = __SOURCE_DIRECTORY__ + "/c4_data/"

type OneFutureModel() =
    let numMsg, Msg = GetRange "Msg"

    let FeatureValue =
        (GetVarArray<float> "FeatureValue" Msg)
            .Attrib(DoNotInfer())

    let WeightPrior, Weight = GetVarWithGaussian "Weight"
    let ThresholdPrior, Threshold = GetVarWithGaussian "Threshold"

    let NoiseVariance =
        (GetVar<float> "NoiseVariance")
            .Attrib(DoNotInfer())

    let RepliedTo = GetVarArray<bool> "RepliedTo" Msg

    do Msg.AddAttribute(Sequential())
    do WeightPrior.AddAttribute(DoNotInfer())
    do ThresholdPrior.AddAttribute(DoNotInfer())

    do
        Variable.ForeachBlock
            Msg
            (fun _ ->
                let score =
                    (FeatureValue.[Msg] * Weight).Named("score")

                let noisyScore =
                    Variable
                        .GaussianFromMeanAndVariance(score, NoiseVariance)
                        .Named("noisyScore")

                RepliedTo.[Msg] <- ((noisyScore >> Threshold).Named("result")))

    do engine.OptimiseForVariables <- [| Weight; Threshold |]

    member this.SetObserved ()=
        numMsg.ObservedValue <- 3
        FeatureValue.ObservedValue <- [| 1; 2; 3 |]
        NoiseVariance.ObservedValue <- 0.5
        RepliedTo.ObservedValue <- [| false; false; true |]
        WeightPrior.ObservedValue <- Gaussian.FromMeanAndVariance(0, 1)
        ThresholdPrior.ObservedValue <-Gaussian.FromMeanAndVariance(0, 100) 

    member this.Infer() =
        let weight = engine.Infer<Gaussian>(Weight)
        let threshold = engine.Infer<Gaussian>(Threshold)

        printfn $"weight post={weight}"
        printfn $"threshold post={threshold}"
        weight, threshold
        

let Infer () =
    let model = OneFutureModel()
    model.SetObserved()
    model.Infer()
