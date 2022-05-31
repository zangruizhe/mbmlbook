module FSharpVersion.Chapter7

open System
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Math

open FSharpVersion.Common


type HonestWorkerModel() =
    // Ranges and indexing
    let numWorkers, Workers = GetRange "Workers"
    let numTweets, Tweets = GetRange "Tweets"
    let numLabels, Labels = GetRange "Labels"

    let Evidence =
        Variable.Bernoulli(0.5).Named("evidence")

    let WorkerJudgmentCount =
        GetArrayVar<int> "WorkerJudgmentCount" Workers

    let WorkerJudgment =
        Range(WorkerJudgmentCount[Workers])
            .Named("WorkerJudgment")

    let WorkerJudgedTweetIndex =
        GetArrayOfArrayVar<int> "WorkerJudgedTweetIndex" WorkerJudgment Workers

    do WorkerJudgedTweetIndex.SetValueRange(Tweets)

    // Truth variable
    let ProbLabelPrior, ProbLabel =
        GetVarWithDist<Vector, Dirichlet> "ProbLabelPrior"

    do ProbLabel.SetValueRange(Labels)

    let TrueLabel =
        GetArrayVar<int> "TrueLabel" Tweets

    do TrueLabel.[Tweets] <- Variable.Discrete(ProbLabel).ForEach(Tweets)

    let numGoldLabels, GoldLabels =
        GetRange "GoldLabels"

    let GoldLabelIndices =
        GetArrayVar<int> "GoldLabelIndices" GoldLabels

    let GoldLabel =
        Variable
            .Subarray(TrueLabel, GoldLabelIndices)
            .Named("GoldLabel")

    let WorkerLabel =
        GetArrayOfArrayVar "WorkerLabel" WorkerJudgment Workers


    // Honest worker
    let AbilityPrior, Ability =
        GetArrayVarWithDst<double, Beta> "Ability" Workers

    let RandomGuessPrior, RandomGuess =
        GetVarWithDist<Vector, Dirichlet> "RandomGuess"

    do
        Variable.ForeachBlock Workers (fun _ ->
            let trueLabels =
                Variable
                    .Subarray(TrueLabel, WorkerJudgedTweetIndex.[Workers])
                    .Named("trueLabelSubarray")

            trueLabels.SetValueRange(Labels)

            Variable.ForeachBlock WorkerJudgment (fun _ ->
                let workersIsCorrect =
                    Variable
                        .Bernoulli(Ability[Workers])
                        .Named("workersIsCorrect")

                Variable.IfBlock
                    workersIsCorrect
                    (fun _ ->
                        let y =
                            WorkerLabel.[Workers][WorkerJudgment]

                        let x = trueLabels.[WorkerJudgment]

                        let labelsEqual =
                            (x == y).Named("labelsEqual")

                        Variable.ConstrainEqualRandom(labelsEqual, Bernoulli(0.9999)) // Add a slight amount of noise due to Infer.NET compiler bug.
                        )
                    (fun _ -> WorkerLabel.[Workers][WorkerJudgment] <- Variable.Discrete(RandomGuess))))

    member this.SetObservation() =
        let Labels =
            [| 15; 16; 17; 18; 19; 300; 400 |]

        let workerJudgedTweetIndex =
            [| [| 1; 2; 3 |]
               [| 2; 3; 4 |]
               [| 0; 1; 2; 3; 4 |] |]

        let workerLabel = // should use label index not label value!
            [| [| 1; 2; 3 |]
               [| 2; 2; 2 |]
               [| 1; 2; 2; 2; 3 |] |]

        let goldLabelsIndices = [| 1; 3 |] // indices which twitter have gold label
        let goldLabels = [| 2; 2 |] // should use label index not label value!

        numGoldLabels.ObservedValue <- goldLabelsIndices.Length
        GoldLabelIndices.ObservedValue <- goldLabelsIndices
        GoldLabel.ObservedValue <- goldLabels

        numLabels.ObservedValue <- Array.length Labels
        numTweets.ObservedValue <- 5
        numWorkers.ObservedValue <- workerLabel.Length

        WorkerJudgmentCount.ObservedValue <-
            workerJudgedTweetIndex
            |> Array.fold (fun state index -> Array.append state [| (Array.length index) |]) Array.empty

        WorkerJudgedTweetIndex.ObservedValue <- workerJudgedTweetIndex

        WorkerLabel.ObservedValue <- workerLabel

        ProbLabelPrior.ObservedValue <- Dirichlet.Uniform(Array.length Labels)
        RandomGuessPrior.ObservedValue <- Dirichlet.Uniform(Array.length Labels)
        AbilityPrior.ObservedValue <- Array.init workerLabel.Length (fun _ -> Beta(2, 1))

    member this.Infer() =
        //        let TrueLabelPost =
//            engine.Infer<Discrete []> TrueLabel
        let TrueLabel =
            Infer<Discrete []> "TrueLabel" engine TrueLabel

        let BackgroundLabelProb =
            Infer<Dirichlet> "BackgroundLabelProb" engine ProbLabel

        let RandomGuessProbability =
            Infer<Dirichlet> "RandomGuessProbability" engine RandomGuess

        let WorkerAbility =
            Infer<Beta []> "WorkerAbility" engine Ability

        ()

let Infer () =
    let honest_model = HonestWorkerModel()
    honest_model.SetObservation()
    honest_model.Infer()
