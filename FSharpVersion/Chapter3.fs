module FSharpVersion.Chapter3

open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Models.Attributes

open FSharpVersion.Common
open MeetingYourMatch
open MeetingYourMatch.Items
open MBMLViews

[<Literal>]
let DataPath = __SOURCE_DIRECTORY__ + "/c3_data/"

let inputs, games =
    let inputs =
        FileUtils.Load<Inputs<TwoPlayerGame>>(DataPath, "Halo2-HeadToHead")

    let games =
        inputs.Games
        |> Seq.filter
            (fun g ->
                g.Players.Contains("Gamer01266") <> true
                && g.Players.Contains("Gamer00296") <> true)
        |> Seq.toArray

    inputs.Players.Remove("Gamer01266") |> ignore
    inputs.Players.Remove("Gamer00296") |> ignore
    inputs, games


let player = Gaussian.FromMeanAndVariance(25, 69.44)
let Skill1 = 120.0
let Skill2 = 100.0
let Sigma1 = 40.0
let Sigma2 = 5.0
let Variance1 = Sigma1 * Sigma1
let Variance2 = Sigma2 * Sigma2

let s1 =
    Gaussian.FromMeanAndVariance(Skill1, Variance1)

let s2 =
    Gaussian.FromMeanAndVariance(Skill2, Variance2)

let Beta = 5.0
let PerformanceVariance = Beta * Beta

type TwoPlayerMessage(performanceVariance: float) =
    //v1
    let skill1Prior = GetVar "JSkillPrior"
    let skill1 = GetVarFromDist "JSkill" skill1Prior

    let skill2Prior = GetVar "FSkillPrior"
    let skill2 = GetVarFromDist "FSkill" skill2Prior

    //    //v2
//    let skill1Prior = Variable.Random(skill1PriorDis).Named("JSkillPrior")
//    let skill1 = Variable.New<float>().Named("JSkill")
//    skill1.SetTo(skill1Prior)

    let player1Performance =
        Variable
            .GaussianFromMeanAndVariance(skill1, performanceVariance)
            .Named("JPerf")

    let player2Performance =
        Variable
            .GaussianFromMeanAndVariance(skill2, performanceVariance)
            .Named("FPerf")

    let player1Wins =
        (player1Performance >> player2Performance)
            .Named("greaterThan")

    member this.SetObserved skill1PriorDis skill2PriorDis outcome =
        skill1Prior.ObservedValue <- skill1PriorDis // also can use for online learn
        skill2Prior.ObservedValue <- skill2PriorDis
        player1Wins.ObservedValue <- outcome

    member this.InferSkill1() =
        let post = engine.Infer<Gaussian>(skill1)
        printfn $"skill1 post={post}"

    member this.InferSkill2() =
        let post = engine.Infer<Gaussian>(skill2)
        printfn $"skill2 post={post}"

type TwoPlayerWithDraw(performanceVariance: float) =
    //v1
    let skill1Prior = GetVar "JSkillPrior"
    let skill1 = GetVarFromDist "JSkill" skill1Prior

    let skill2Prior = GetVar "FSkillPrior"
    let skill2 = GetVarFromDist "FSkill" skill2Prior

    let drawMarginPrior = GetVar "drawMarginPrior"

    let drawMargin =
        GetVarFromDist "drawMargin" drawMarginPrior

    //    //v2
//    let skill1Prior = Variable.Random(skill1PriorDis).Named("JSkillPrior")
//    let skill1 = Variable.New<float>().Named("JSkill")
//    skill1.SetTo(skill1Prior)

    let player1Performance =
        Variable
            .GaussianFromMeanAndVariance(skill1, performanceVariance)
            .Named("JPerf")

    let player2Performance =
        Variable
            .GaussianFromMeanAndVariance(skill2, performanceVariance)
            .Named("FPerf")

    let diff =
        (player1Performance - player2Performance)
            .Named("diff")

    let outcome =
        Variable.DiscreteUniform(3).Named("outcome")

    do Variable.ConstrainTrue(drawMargin >> 0.0)
    do using (Variable.Case(outcome, 0)) (fun _ -> Variable.ConstrainTrue(diff >> drawMargin))
    do using (Variable.Case(outcome, 1)) (fun _ -> Variable.ConstrainBetween(diff, -drawMargin, drawMargin))
    do using (Variable.Case(outcome, 2)) (fun _ -> Variable.ConstrainTrue(diff << -drawMargin))

    member this.SetObserved skill1PriorDis skill2PriorDis marginPriorDis outcomePrior =
        skill1Prior.ObservedValue <- skill1PriorDis // also can use for online learn
        skill2Prior.ObservedValue <- skill2PriorDis
        drawMarginPrior.ObservedValue <- marginPriorDis
        outcome.ObservedValue <- outcomePrior

    member this.Infer() =
        let sk1 = engine.Infer<Gaussian>(skill1)
        let sk2 = engine.Infer<Gaussian>(skill2)
        let drawMargin = engine.Infer<Gaussian>(drawMargin)

        printfn $"skill1 post={sk1}"
        printfn $"skill2 post={sk2}"
        printfn $"drawMargin post={drawMargin}"
        sk1, sk2, drawMargin

type MultiPlayerWithDraw(performanceVariance: float) =

    let numPlayer, Player = GetRange "Player"
    let numGamePlayer, gamePlayer = GetRange "gamePlayer"

    let skillsPrior =
        GetArrayVar<Gaussian> "skillPriors" Player

    let skills = GetArrayVar<float> "skills" Player
    do skills.[Player] <- Variable.GaussianFromMeanAndVariance(Variable.Random(skillsPrior.[Player]), 1.44)

    let performanceVariance =
        Variable
            .Observed(performanceVariance)
            .Named("performanceVariance")

    let drawMarginPrior = GetVar "drawMarginPrior"

    let drawMargin =
        GetVarFromDist "drawMargin" drawMarginPrior

    do Variable.ConstrainTrue(drawMargin >> 0.0)

    let playerIndices =
        GetArrayVar<int> "playerIndices" gamePlayer

    let performances =
        GetArrayVar<double> "performance" gamePlayer

    let gameSkills =
        Variable
            .Subarray(skills, playerIndices)
            .Named("gameSkills")

    let scores =
        (GetArrayVar<int> "scores" gamePlayer)
            .Attrib(DoNotInfer())

    do performances.[gamePlayer] <- Variable.GaussianFromMeanAndVariance(gameSkills.[gamePlayer], performanceVariance)

    do
        let gp = Variable.ForEach(gamePlayer)

        do
            using
                (Variable.If(gp.Index >> 0))
                (fun _ ->
                    let first = performances.[gp.Index - 1]
                    let second = performances.[gp.Index]
                    let diff = (first - second).Named("diff")

                    Variable.IfBlock
                        (scores.[gp.Index - 1] == scores.[gp.Index])
                        (fun _ -> Variable.ConstrainBetween(diff, -drawMargin, drawMargin))
                        (fun _ -> Variable.ConstrainTrue(diff >> drawMargin)))

        gp.Dispose()

    member this.SetObserved() =

        let p1 = Gaussian(120, 400)
        let p2 = Gaussian(100, 1600)
        let p3 = Gaussian(140, 1600)
        drawMarginPrior.ObservedValue <- Gaussian.PointMass(0)
        numPlayer.ObservedValue <- 3
        skillsPrior.ObservedValue <- [| p1; p2; p3 |]
        numGamePlayer.ObservedValue <- 3
        playerIndices.ObservedValue <- [| 0; 1; 2 |]
        scores.ObservedValue <- [| 2; 1; 0 |]

    member this.Infer() =
        let drawMargin = engine.Infer<Gaussian>(drawMargin)
        let skillsPost = engine.Infer<Gaussian []>(skills)

        printfn $"drawMargin post=%A{drawMargin}"
        printfn $"skills post=%A{skillsPost}"
        skillsPost, drawMargin

let GetResult (game: TwoPlayerGame) =
    match game.Player1Score = game.Player2Score, game.Player1Score < game.Player2Score with
    | true, _ -> 1
    | _, true -> 2
    | _ -> 0

let MultiGames () =

    let prior = inputs.TrueSkillPriors
    let plays = inputs.Players
    let skillPrior = inputs.SkillPrior

    let mutable margins = [ prior.DrawMargin ]
    let mutable predictions = List.empty

    let mutable post =
        prior.Skills.Keys
        |> Seq.map (fun k -> (k, [ prior.Skills.[k] ]))
        |> Map.ofSeq

    let model =
        TwoPlayerWithDraw(inputs.TrueSkillParameters.PerformanceVariance)

    games
    |> Array.iter
        (fun g ->
            let p1 = post.[g.Player1]
            let p2 = post.[g.Player2]
            let margin = List.head margins

            model.SetObserved p1.Head p2.Head margin (int g.Outcome)
            let p1p, p2p, mp = model.Infer()
            post <- Map.add g.Player1 (p1p :: p1) post
            post <- Map.add g.Player2 (p2p :: p2) post
            margins <- mp :: margins)

    ()

let OneGame () =
    let model = TwoPlayerWithDraw(17.36)
    model.SetObserved player player (Gaussian.FromMeanAndVariance(1, 10)) 2
    model.Infer()

let SimpleOneGame () =
    let model = TwoPlayerMessage(PerformanceVariance)
    model.SetObserved s1 s2 true
    model.InferSkill1()

let MultiPlayerGame () =
    let model = MultiPlayerWithDraw(PerformanceVariance)
    model.SetObserved()
    model.Infer()

let Infer () =
    //    // game without draw
//    SimpleOneGame()
//
//    // game with draw
//    OneGame() |> ignore
//
//    // whole game
//    MultiGames()

    // multi player game
    MultiPlayerGame()
