module FSharpVersion.Chapter2

open Microsoft.ML.Probabilistic.Distributions
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

let ProbabilityOfGuess = 0.2
let ProbabilityOfNotMistake = 0.9
let ProbabilityOfSkillTrue = 0.5

module Factor =
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


    let NoisyAllTrue
        (variableArray: VariableArray<bool>)
        (probNoMistake: Variable<float>)
        (probGuess: Variable<float>)
        : Variable<bool> =
        let hasSkills =
            Variable.AllTrue(variableArray).Named("hasSkills")

        AddNoise hasSkills probNoMistake probGuess

module NoisyAndModel =
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
            .Array<VariableArray<bool>, bool [] []>(Variable.Array<bool>(Skills), Questions)
            .Named("skillsQuestionsMask")
            .Attrib(DoNotInfer())

    let skill =
        Variable
            .Array<VariableArray<bool>, bool [] []>(Variable.Array<bool>(Skills), People)
            .Named("skill")

    skill.[People][Skills] <- Variable
        .Bernoulli(probSkillTrue[Skills])
        .ForEach(People)

    let isCorrect =
        Variable
            .Array<VariableArray<bool>, bool [] []>(Variable.Array<bool>(Questions), People)
            .Named("isCorrect")

    let Construct (inputs: Inputs) obs_true_skill obs_correct =

        let ConstructNoisyFactor () =
            Variable.ForeachBlock People (fun _ ->
                Variable.ForeachBlock Questions (fun _ ->
                    let relevantSkills =
                        Variable
                            .Subarray(skill[People], skillsNeeded[Questions])
                            .Named("relevantSkills")

                    let hasSkills =
                        Variable
                            .AllTrue(relevantSkills)
                            .Named("hasSkills")

                    isCorrect.[People][Questions] <- Factor.AddNoise
                                                         hasSkills
                                                         probNotMistake[Questions]
                                                         probGuess[Questions]))

            ()

        // add observe
        let SetObserved (inputs: Inputs) obs_correct obs_true_skill =
            //        skillsQuestionsMask.ObservedValue <- inputs.Quiz.SkillsQuestionsMask

            numPeople.ObservedValue <- inputs.NumberOfPeople
            numSkills.ObservedValue <- inputs.Quiz.NumberOfSkills
            numQuestions.ObservedValue <- inputs.Quiz.NumberOfQuestions
            numSkillsForEachQuestion.ObservedValue <- inputs.Quiz.NumberSkillsForQuestion

            skillsNeeded.ObservedValue <- inputs.Quiz.SkillsForQuestion

            match obs_correct, obs_true_skill with
            | true, false ->
                isCorrect.ObservedValue <- inputs.IsCorrect
                skill.ClearObservedValue()
            | false, true ->
                skill.ObservedValue <- inputs.StatedSkills
                isCorrect.ClearObservedValue()
            | true, true ->
                skill.ObservedValue <- inputs.StatedSkills
                isCorrect.ObservedValue <- inputs.IsCorrect
            | false, false ->
                //            skill.ClearObservedValue()
                //            isCorrect.ClearObservedValue()
                failwith "should not come to here"

                //            probGuess.ClearObservedValue()
                probGuess.ObservedValue <- Array.init numQuestions.ObservedValue (fun _ -> ProbabilityOfGuess)

                //            probNotMistake.ClearObservedValue()
                probNotMistake.ObservedValue <- Array.init numQuestions.ObservedValue (fun _ -> ProbabilityOfNotMistake)

                //            probSkillTrue.ClearObservedValue()
                probSkillTrue.ObservedValue <- Array.init numSkills.ObservedValue (fun _ -> ProbabilityOfSkillTrue)

            ()

        let ClearObservedVariables () =
            skillsNeeded.ClearObservedValue()
            skillsQuestionsMask.ClearObservedValue()
            skill.ClearObservedValue()
            isCorrect.ClearObservedValue()
            numPeople.ClearObservedValue()
            numQuestions.ClearObservedValue()
            numSkills.ClearObservedValue()
            numSkillsForEachQuestion.ClearObservedValue()
            probGuess.ClearObservedValue()
            probNotMistake.ClearObservedValue()
            probSkillTrue.ClearObservedValue()

            ()

        ConstructNoisyFactor()
        SetObserved inputs obs_correct obs_true_skill

        ()

    let InferSkills () =
        let posterior =
            engine.Infer<Bernoulli>(skill)

        printfn $"skill posterior={posterior}"

    let InferIsCorrect () =
        let posterior =
            engine.Infer<Bernoulli>(isCorrect)

        printfn $"isCorrect posterior={posterior}"



let Infer () =
    printfn $"Chapter2 start"

    let input3 =
        FileUtils.Load<Inputs>(DataPath, "Toy3")

    NoisyAndModel.Construct input3 false true

    NoisyAndModel.InferSkills()
