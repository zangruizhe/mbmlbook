module FSharpVersion.Chapter3

open Microsoft.FSharp.Quotations
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Models.Attributes

open FSharpVersion.Common

let  Skill1 = 120.0
let  Skill2 = 100.0
let  Sigma1 = 40.0
let  Sigma2 = 5.0
let  Variance1 = Sigma1 * Sigma1
let  Variance2 = Sigma2 * Sigma2

let s1 = Gaussian.FromMeanAndVariance(Skill1, Variance1)
let s2 = Gaussian.FromMeanAndVariance(Skill2, Variance2)
let  Beta = 5.0
let  PerformanceVariance = Beta * Beta

type TwoPlayerMessage(performanceVariance:float) =
    //v1
    let skill1Prior = GetGaussianVar "JSkillPrior"
    let skill1 = GetVarFromGaussian "JSkill" skill1Prior
    
    let skill2Prior = GetGaussianVar "FSkillPrior"
    let skill2 = GetVarFromGaussian "FSkill" skill2Prior
    
//    //v2
//    let skill1Prior = Variable.Random(skill1PriorDis).Named("JSkillPrior")
//    let skill1 = Variable.New<float>().Named("JSkill")
//    skill1.SetTo(skill1Prior)
    
    let player1Performance = Variable.GaussianFromMeanAndVariance(skill1, performanceVariance).Named("JPerf")
    let player2Performance = Variable.GaussianFromMeanAndVariance(skill2, performanceVariance).Named("FPerf")
    
    let player1Wins = (player1Performance >> player2Performance).Named("greaterThan")
    
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
        
type TwoPlayerWithDraw(performanceVariance:float) =
    //v1
    let skill1Prior = GetGaussianVar "JSkillPrior"
    let skill1 = GetVarFromGaussian "JSkill" skill1Prior
    
    let skill2Prior = GetGaussianVar "FSkillPrior"
    let skill2 = GetVarFromGaussian "FSkill" skill2Prior
    
    let drawMarginPrior = GetGaussianVar "drawMarginPrior"
    let drawMargin = GetVarFromGaussian "drawMargin" drawMarginPrior
    
//    //v2
//    let skill1Prior = Variable.Random(skill1PriorDis).Named("JSkillPrior")
//    let skill1 = Variable.New<float>().Named("JSkill")
//    skill1.SetTo(skill1Prior)
    
    let player1Performance = Variable.GaussianFromMeanAndVariance(skill1, performanceVariance).Named("JPerf")
    let player2Performance = Variable.GaussianFromMeanAndVariance(skill2, performanceVariance).Named("FPerf")
    let diff = (player1Performance - player2Performance).Named("diff")
    let outcome = Variable.DiscreteUniform(3).Named("outcome")
    
    member this.WinLoseDraw() =
        do using (Variable.Case(outcome, 0)) (fun _ -> Variable.ConstrainTrue(diff >> drawMargin))
        do using (Variable.Case(outcome, 1)) (fun _ -> Variable.ConstrainBetween(diff, -drawMargin, drawMargin))
        do using (Variable.Case(outcome, 2)) (fun _ -> Variable.ConstrainTrue(diff << -drawMargin))
    
    member this.Construct () =
        Variable.ConstrainTrue(drawMargin >> 0.0)
        this.WinLoseDraw()
        
    member this.SetObserved skill1PriorDis skill2PriorDis marginPriorDis outcomePrior= 
        skill1Prior.ObservedValue <- skill1PriorDis // also can use for online learn
        skill2Prior.ObservedValue <- skill2PriorDis
        drawMarginPrior.ObservedValue <- marginPriorDis
        outcome.ObservedValue <- outcomePrior
        
    member this.Infer() =
        printfn $"skill1 post={engine.Infer<Gaussian>(skill1)}"
        printfn $"skill2 post={engine.Infer<Gaussian>(skill2)}"
        printfn $"drawMargin post={engine.Infer<Gaussian>(drawMargin)}"

let Infer() =
    // game without draw
    let model = TwoPlayerMessage(PerformanceVariance)
    model.SetObserved s1 s2 true
    model.InferSkill1()
    
    
    // game with draw
    let player = Gaussian.FromMeanAndVariance(25, 69.44)
    let model = TwoPlayerWithDraw(17.36)
    model.Construct()
    model.SetObserved player player  (Gaussian.FromMeanAndVariance(1, 10)) 2
    model.Infer()
