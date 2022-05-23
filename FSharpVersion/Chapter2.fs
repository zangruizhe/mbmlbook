module FSharpVersion.Chapter2

open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Models.Attributes

open MBMLViews
open AssessingPeoplesSkills
open FSharpVersion.Common

[<Literal>]
let DataPath =
    __SOURCE_DIRECTORY__
    + "/../src/2. Assessing Peoples Skills/Data/"

let AddNoise (hasSkills: Variable<bool>) (probNoMistake: Variable<float>) (probGuess: Variable<float>) =
    let mutable noisyAllTrue =
        Variable.New<bool>().Named("NoisyAllTrueGated")

    Variable.IfBlock
        hasSkills
        (fun _ ->
            noisyAllTrue.SetTo(
                Variable
                    .Bernoulli(probNoMistake)
                    .Named("probIfAllTrue")
            ))
        (fun _ ->
            noisyAllTrue.SetTo(
                Variable
                    .Bernoulli(probGuess)
                    .Named("probIfNotAllTrue")
            ))

    noisyAllTrue


let NoisyAllTrue (variableArray: VariableArray<bool>) (probNoMistake: Variable<float>) (probGuess: Variable<float>): Variable<bool> =
    let hasSkills =
        Variable.AllTrue(variableArray).Named("hasSkills")
    AddNoise hasSkills probNoMistake probGuess

let ToyWith3Q2S () =
    let numPeople, People = GetRange "People"

    let numSkills, Skills = GetRange "Skills"

    let numQuestions, Questions =
        GetRange "Questions"

    let probGuess =
        GetVarArray<float> "probGuess" Questions

    let probNotMistake =
        GetVarArray<float> "probNoMistake" Questions

    let probSkillTrue =
        GetVarArray<float> "probSkillTrue" Skills

    let numSkillsForEachQuestion =
        (GetVarArray<int> "numSkillsForQuestions" Questions)
            .Attrib(DoNotInfer())

    let QuestionsSkills =
        Range(numSkillsForEachQuestion[Questions])
            .Named("questionsXskills")

    let skillsNeeded =
        Variable
            .Array<VariableArray<int>, int [] []>(Variable.Array<int>(QuestionsSkills), Questions)
            .Named("skillsNeeded")
            .Attrib(DoNotInfer())

    let skillsQuestionsMask =
        Variable
            .Array<VariableArray<bool>, int [] []>(Variable.Array<bool>(Skills), Questions)
            .Named("skillsQuestionsMask")
            .Attrib(DoNotInfer())

    let skill =
        Variable
            .Array<VariableArray<bool>, int [] []>(Variable.Array<bool>(Skills), People)
            .Named("skill")

    skill.[People][Skills] <- Variable
        .Bernoulli(probSkillTrue[Skills])
        .ForEach(People)

    let isCorrect =
        Variable
            .Array<VariableArray<bool>, int [] []>(Variable.Array<bool>(Questions), People)
            .Named("isCorrect")

    ()


let Infer () =
    let input3 =
        FileUtils.Load<Inputs>(DataPath, "Toy3")

    printfn $"%A{input3}"
