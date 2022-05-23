module FSharpVersion.Chapter1

open System
open Microsoft.FSharp.Quotations
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.FSharp

let Grey = 0.3
let Auburn = 0.7

let RevolverGivenGrey = 0.9
let RevolverGivenAuburn = 0.2

let DaggerGivenGrey = 0.1
let DaggerGivenAuburn = 0.8

let HairGivenGrey = 0.5
let HairGivenAuburn = 0.05

let engine = InferenceEngine()
engine.ShowFactorGraph <- true;

let mudder_is_grey = Variable.Bernoulli(Grey)
let weapon_is_revolver = Variable.New<bool>().Named("weapon_is_revolver")

Variable.IfBlock mudder_is_grey
            (fun _ -> weapon_is_revolver.SetTo(Variable.Bernoulli(RevolverGivenGrey)))
            (fun _ -> weapon_is_revolver.SetTo(Variable.Bernoulli(RevolverGivenAuburn)))

let Infer () =
    weapon_is_revolver.ObservedValue <- true
    let mudder_is_grey_posterior = engine.Infer<Bernoulli>(mudder_is_grey)
    printfn $"mudder_is_grey_posterior={mudder_is_grey_posterior}"
    printfn $"mudder_is_grey_posterior(exp(log))={Math.Exp(mudder_is_grey_posterior.GetLogProbTrue())}"
