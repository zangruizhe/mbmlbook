module FSharpVersion.Chapter1

open System
open FSharpVersion.Common
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

let CreateBase () =
    let mudder_is_grey =
        Variable.Bernoulli(Grey)

    mudder_is_grey

let private infer mudder_is_grey =
    let mudder_is_grey_posterior =
        engine.Infer<Bernoulli>(mudder_is_grey)

    printfn $"mudder_is_grey_posterior={mudder_is_grey_posterior}"
    printfn $"mudder_is_grey_posterior(exp(log))={Math.Exp(mudder_is_grey_posterior.GetLogProbTrue())}"

let CreateWeapon mudder_is_grey =
    let weapon_is_revolver =
        Variable.New<bool>().Named("weapon_is_revolver")

    Variable.IfBlock
        mudder_is_grey
        (fun _ -> weapon_is_revolver.SetTo(Variable.Bernoulli(RevolverGivenGrey)))
        (fun _ -> weapon_is_revolver.SetTo(Variable.Bernoulli(RevolverGivenAuburn)))

    weapon_is_revolver.ObservedValue <- true

let CreateHair mudder_is_grey =
    let find_hair =
        Variable.New<bool>().Named("find_hair")

    Variable.IfBlock
        mudder_is_grey
        (fun _ -> find_hair.SetTo(Variable.Bernoulli(HairGivenGrey)))
        (fun _ -> find_hair.SetTo(Variable.Bernoulli(HairGivenAuburn)))

    find_hair.ObservedValue <- true

let Infer () =
    printfn $"#####   weapon is revolver"

    CreateBase()
    |> fun prior ->
        CreateWeapon prior
        infer prior

    printfn $"#####   weapon is revolver and find hair"

    CreateBase()
    |> fun prior ->
        CreateWeapon prior
        CreateHair prior
        infer prior
