module FSharpVersion.Chapter4

open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Models.Attributes
open Microsoft.ML.Probabilistic.Collections

open System.Linq
open FSharpVersion.Common
open UnclutteringYourInbox

[<Literal>]
let DataPath = __SOURCE_DIRECTORY__ + "/c4_data/"

let csharp_runner = ModelRunner()
ModelRunner.LoadAllInputFiles(DataPath)

type OneFutureModel() =
    let numMsg, Msg = GetRange "Msg"

    let FeatureValue =
        (GetArrayVar<float> "FeatureValue" Msg)
            .Attrib(DoNotInfer())

    let WeightPrior, Weight = GetVarWithGaussian "Weight"
    let ThresholdPrior, Threshold = GetVarWithGaussian "Threshold"

    let NoiseVariance =
        (GetVar<float> "NoiseVariance")
            .Attrib(DoNotInfer())

    let RepliedTo = GetArrayVar<bool> "RepliedTo" Msg

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

    member this.SetObserved() =
        let input =
            ModelRunner.OneFeature.Noise.InputsCollection.Inputs.[0]

        let train = input.TrainAndValidation.Instances

        numMsg.ObservedValue <- train.Count

        FeatureValue.ObservedValue <-
            train
            |> Seq.map (fun m -> m.FeatureValues.First().Value)
            |> Seq.toArray

        NoiseVariance.ObservedValue <- 10

        WeightPrior.ObservedValue <- Gaussian.FromMeanAndVariance(0, 1)
        ThresholdPrior.ObservedValue <- Gaussian.FromMeanAndVariance(0, 10)
        RepliedTo.ObservedValue <- train |> Seq.map (fun m -> m.Label) |> Seq.toArray

    member this.Infer() =
        let weight = engine.Infer<Gaussian>(Weight)
        let threshold = engine.Infer<Gaussian>(Threshold)

        printfn $"weight post={weight}"
        printfn $"threshold post={threshold}"
        weight, threshold

type ReplyToModel() =
    // define variable
    let numMsg, Emails = GetRange "Msg"
    let numFeatures, Features = GetRange "Features"

    let FeatureValue =
        GetArrayOfArrayVar<float> "FeatureValue" Features Emails

    let WeightPrior, Weight = GetGaussianArray "Weight" Features
    let ThresholdPrior, Threshold = GetVarWithGaussian "Threshold"
    let NoiseVariance = GetVar<float> "NoiseVariance"
    let RepliedTo = GetArrayVar<bool> "RepliedTo" Emails

    do
        Variable.ForeachBlock
            Emails
            (fun _ ->
                let FeatureScore =
                    GetArrayVar<float> "FeatureScore" Features

                FeatureScore.[Features] <-
                    FeatureValue.[Emails].[Features]
                    * Weight.[Features]

                let score =
                    Variable.Sum(FeatureScore).Named("score")

                let noisyScore =
                    Variable
                        .GaussianFromMeanAndVariance(score, NoiseVariance)
                        .Named("noisyScore")

                RepliedTo.[Emails] <- (noisyScore >> Threshold))


    // simple predict graph
    do Emails.AddAttribute(Sequential())

    do AddDoNotInfer numMsg
    do AddDoNotInfer numFeatures
    do AddDoNotInfer WeightPrior
    do AddDoNotInfer ThresholdPrior
    do AddDoNotInfer FeatureValue
    do AddDoNotInfer NoiseVariance

    do engine.OptimiseForVariables <- [| Weight; Threshold |]

    member this.SetObserved() =
        let input =
            ModelRunner.CombiningFeatures.Separate.InputsCollection.Inputs.[0]

        let train = input.TrainAndValidation.Instances

        numMsg.ObservedValue <- train.Count
        numFeatures.ObservedValue <- input.FeatureSet.FeatureVectorLength
        NoiseVariance.ObservedValue <- 10

        RepliedTo.ObservedValue <- train |> Seq.map (fun m -> m.Label) |> Seq.toArray
        ThresholdPrior.ObservedValue <- Gaussian.FromMeanAndVariance(0, 10)

        WeightPrior.ObservedValue <-
            Array.init input.FeatureSet.FeatureVectorLength (fun _ -> Gaussian.FromMeanAndVariance(0, 1))

        let features =
            Array.init
                train.Count
                (fun i ->
                    let row =
                        Array.init input.FeatureSet.FeatureVectorLength (fun j -> 0.0)

                    train.[i].FeatureValues
                    |> Seq.iter
                        (fun featureBucketValue ->
                            let index =
                                train.[i]
                                    .FeatureSet.FeatureBuckets.FindIndex(fun key -> key = featureBucketValue.Key)

                            match index < 0 with
                            | true -> failwith "can not go to here"
                            | false -> row.[index] <- featureBucketValue.Value)

                    row)

        FeatureValue.ObservedValue <- features


    member this.Infer() =
        let weight = engine.Infer<Gaussian []>(Weight)
        let threshold = engine.Infer<Gaussian>(Threshold)

        printfn $"weight post=%A{weight}"
        printfn $"threshold post={threshold}"
        weight, threshold


let Infer () =
    //    let model = OneFutureModel()
//    model.SetObserved()
//    model.Infer() |> ignore

    let model = ReplyToModel()
    model.SetObserved()
    model.Infer() |> ignore
    ()
