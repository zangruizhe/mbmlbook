module FSharpVersion.Chapter2

open Microsoft.FSharp.Quotations
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Models.Attributes

open MBMLViews
open MBMLCommon
open AssessingPeoplesSkills
open FSharpVersion.Common

[<Literal>]
let DataPath = __SOURCE_DIRECTORY__ + "/c2_data/"

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

    let probNotMistake =
        GetArrayVar<float> "probNoMistake" Questions

    let probGuess =
        GetArrayVar<float> "probGuess" Questions
        
    let probSkillTrue =
        GetArrayVar<float> "probSkillTrue" Skills

    let numSkillsForEachQuestion =
        (GetArrayVar<int> "numSkillsForQuestions" Questions)
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

    let SetObserved (inputs: Inputs) obs_correct obs_true_skill =
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
        | false, false -> failwith "should not come to here"


        probGuess.ObservedValue <- Array.init numQuestions.ObservedValue (fun _ -> ProbabilityOfGuess)
        probNotMistake.ObservedValue <- Array.init numQuestions.ObservedValue (fun _ -> ProbabilityOfNotMistake)
        probSkillTrue.ObservedValue <- Array.init numSkills.ObservedValue (fun _ -> ProbabilityOfSkillTrue)

    let Construct (inputs: Inputs) obs_true_skill obs_correct =
        ConstructNoisyFactor()
        SetObserved inputs obs_correct obs_true_skill

    let InferSkills () =
        let posterior =
            engine.Infer<Bernoulli [] []>(skill)

        posterior
        |> Array.iteri (fun i post ->
            printfn $"people[{i}]skill posterior=%A{post}"

            post
            |> Array.iteri (fun j p -> printfn $"\t\tpeople[{i}] skill[{j}] posterior=%A{p.LogOdds}"))

    let InferIsCorrect () =
        let posterior =
            engine.Infer<Bernoulli [] []>(isCorrect)

        printfn $"isCorrect posterior=%A{posterior}"

module Unrolled =
    let inputs =
        FileUtils.Load<Inputs>(DataPath, "Toy3")

    let numPeople, People = GetRange "People"
    let numSkills, Skills = GetRange "Skills"

    let numQuestions, Questions =
        GetRange "Questions"

    let probNotMistake =
        GetArrayVar<float> "probNoMistake" Questions

    let probGuess =
        GetArrayVar<float> "probGuess" Questions

    let probSkillTrue =
        GetArrayVar<float> "probSkillTrue" Skills

    let numSkillsForEachQuestion =
        (GetArrayVar<int> "numSkillsForQuestions" Questions)
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

    let Get2D i j =
        Array.init i (fun _ -> Array.init j (fun _ -> Variable.New<bool>()))

    let mutable unrolledSkills =
        Get2D inputs.NumberOfPeople inputs.Quiz.NumberOfSkills

    unrolledSkills
    |> Array.iteri (fun p cs ->
        cs
        |> Array.iteri (fun s c ->
            unrolledSkills.[p][s] <- Variable
                .Bernoulli(ProbabilityOfSkillTrue)
                .Named(inputs.Quiz.SkillNames[s] + $"_{p}")))

    let mutable isCorrect =
        Get2D inputs.NumberOfPeople inputs.Quiz.NumberOfQuestions

    isCorrect
    |> Array.iteri (fun p cs ->
        cs
        |> Array.iteri (fun q c ->
            isCorrect.[p][q] <- Variable
                .New<bool>()
                .Named("isCorrect" + $"_{p}_{q}")))

    let ClearObservedVariables () =
        skillsNeeded.ClearObservedValue()
        skillsQuestionsMask.ClearObservedValue()
        //        skill.ClearObservedValue()
//        isCorrect.ClearObservedValue()
        numPeople.ClearObservedValue()
        numQuestions.ClearObservedValue()
        numSkills.ClearObservedValue()
        numSkillsForEachQuestion.ClearObservedValue()
        probGuess.ClearObservedValue()
        probNotMistake.ClearObservedValue()
        probSkillTrue.ClearObservedValue()

    let ConstructNoisyFactor () = ()
    //        Variable.ForeachBlock People (fun _ ->
//            Variable.ForeachBlock Questions (fun _ ->
//                let relevantSkills =
//                    Variable
//                        .Subarray(skill[People], skillsNeeded[Questions])
//                        .Named("relevantSkills")
//
//                let hasSkills =
//                    Variable
//                        .AllTrue(relevantSkills)
//                        .Named("hasSkills")
//
//                isCorrect.[People][Questions] <- Factor.AddNoise
//                                                     hasSkills
//                                                     probNotMistake[Questions]
//                                                     probGuess[Questions]))

    let SetObserved (inputs: Inputs) obs_correct obs_true_skill =
        numPeople.ObservedValue <- inputs.NumberOfPeople
        numSkills.ObservedValue <- inputs.Quiz.NumberOfSkills
        numQuestions.ObservedValue <- inputs.Quiz.NumberOfQuestions
        numSkillsForEachQuestion.ObservedValue <- inputs.Quiz.NumberSkillsForQuestion
        skillsNeeded.ObservedValue <- inputs.Quiz.SkillsForQuestion
        probGuess.ObservedValue <- Array.init numQuestions.ObservedValue (fun _ -> ProbabilityOfGuess)
        probNotMistake.ObservedValue <- Array.init numQuestions.ObservedValue (fun _ -> ProbabilityOfNotMistake)
        probSkillTrue.ObservedValue <- Array.init numSkills.ObservedValue (fun _ -> ProbabilityOfSkillTrue)

        match obs_correct, obs_true_skill with
        | true, false ->
            isCorrect
            |> Array.iteri (fun p cs ->
                cs
                |> Array.iteri (fun q c ->
                    c.ObservedValue <- inputs.IsCorrect.[p][q]

                    let skills =
                        inputs.Quiz.SkillsForQuestion[q]

                    match Array.length skills with
                    | 1 ->
                        (isCorrect.[p][q])
                            .SetTo(
                                Factor.AddNoise
                                    (unrolledSkills.[p][skills[0]]) // (probNotMistake[q]) (probGuess[q])
                                    (Variable.Observed(ProbabilityOfNotMistake))
                                    (Variable.Observed(ProbabilityOfGuess))
                            )
                    | 2 ->
                        let hasSkills =
                            (unrolledSkills.[p][skills[0]]
                             &&& unrolledSkills.[p][skills[1]])
                                .Named("hasSkills" + $"_q{q}")

                        (isCorrect.[p][q])
                            .SetTo(
                                Factor.AddNoise
                                    hasSkills //(probNotMistake[q]) (probGuess[q])
                                    (Variable.Observed(ProbabilityOfNotMistake))
                                    (Variable.Observed(ProbabilityOfGuess))
                            )
                    | _ -> failwith "can not go to here"))

        | _ -> failwith "should not come to here"

    let Construct (inputs: Inputs) obs_true_skill obs_correct =
        ConstructNoisyFactor()
        SetObserved inputs obs_correct obs_true_skill

    let InferSkills () =

        //        engine.Compiler.GivePriorityTo(typeof<ReplicateOp_NoDivide>)
//        let p = 0
//        let s = 0
//        let x = unrolledSkills.[p][s]
//        let post = engine.Infer<Bernoulli>(x)
//        printfn $"\t\tpeople[{p}] skill[{s}] posterior=%A{post.GetLogProbTrue()}"

        for p in 0 .. inputs.NumberOfPeople - 1 do
            (for s in 0 .. inputs.Quiz.NumberOfSkills - 1 do
                let post =
                    engine.Infer<Bernoulli>(unrolledSkills.[p][s])

                printfn $"\t\tpeople[{p}] skill[{s}] posterior=%A{post}"
                printfn $"\t\tpeople[{p}] skill[{s}] posterior=%A{post.GetLogProbTrue()}")

        ()

//    let InferIsCorrect () =
//        let posterior =
//            engine.Infer<Bernoulli [] []>(isCorrect)
//
//        printfn $"isCorrect posterior=%A{posterior}"

let outputter =
    Outputter.GetOutputter(Contents.ChapterName)

let runner = ModelRunner(outputter)

let CSharpVersion () =
    let model = Models.UnrolledModel()
    model.Name <- "ThreeQuestions"
    model.ProbabilityOfGuess <- 0.2
    model.ProbabilityOfNotMistake <- 0.9
    model.ProbabilityOfSkillTrue <- 0.5
    model.ShowFactorGraph <- false
    let experiment = Experiment()
    experiment.Inputs <- FileUtils.Load<Inputs>(DataPath, "Toy3")
    experiment.Model <- model
    runner.AnnounceAndRun(experiment, Contents.S2TestingOutTheModel.NumberedName)
    
let SampleFromModel () =
    let input = FileUtils.Load<Inputs>(DataPath, "InputData")
    let model = Models.NoisyAndModel()
    model.Name <- "Original"
    model.ProbabilityOfGuess <- 0.2
    model.ProbabilityOfNotMistake <- 0.9
    model.ProbabilityOfSkillTrue <- 0.5
    model.ShowFactorGraph <- false
    let experiment = Experiment()
    experiment.Inputs <- input
    experiment.Model <- model
    
//    let fake_input_from_real_skill_data = experiment.Model.SampleFromModel(input, input.NumberOfPeople, 0.5, true)
    let fake_input_from_real_skill_data = experiment.Model.SampleFromModel(input, input.NumberOfPeople, 0.5, false)
    NoisyAndModel.Construct fake_input_from_real_skill_data false true
    NoisyAndModel.InferSkills()
    ()
    
    
module LearnedNoisyAndModel =
    let numPeople, People = GetRange "People"
    let numSkills, Skills = GetRange "Skills"

    let numQuestions, Questions =
        GetRange "Questions"

    let probNotMistake =
        GetArrayVar<float> "probNoMistake" Questions

    let probGuess =
        GetArrayVar<float> "probGuess" Questions
        
    Variable.ForeachBlock Questions (fun _ -> probGuess[Questions] <- Variable.Beta(2.5, 7.5).Named("GuessPrior"))
//    Variable.ForeachBlock Questions (fun _ -> probGuess[Questions].SetTo(Variable.Beta(2.5, 7.5).Named("GuessPrior")))
//    probGuess.[People][Skills] <- Variable
//        .Bernoulli(probSkillTrue[Skills])
//        .ForEach(People)

    let probSkillTrue =
        GetArrayVar<float> "probSkillTrue" Skills
        

    let numSkillsForEachQuestion =
        (GetArrayVar<int> "numSkillsForQuestions" Questions)
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

    let SetObserved (inputs: Inputs) obs_correct obs_true_skill =
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
        | false, false -> failwith "should not come to here"


        probGuess.ObservedValue <- Array.init numQuestions.ObservedValue (fun _ -> ProbabilityOfGuess)
        probNotMistake.ObservedValue <- Array.init numQuestions.ObservedValue (fun _ -> ProbabilityOfNotMistake)
        probSkillTrue.ObservedValue <- Array.init numSkills.ObservedValue (fun _ -> ProbabilityOfSkillTrue)

    let Construct (inputs: Inputs) obs_true_skill obs_correct =
        ConstructNoisyFactor()
        SetObserved inputs obs_correct obs_true_skill

    let InferSkills () =
        let posterior =
            engine.Infer<Bernoulli [] []>(skill)

        posterior
        |> Array.iteri (fun i post ->
            printfn $"people[{i}]skill posterior=%A{post}"

            post
            |> Array.iteri (fun j p -> printfn $"\t\tpeople[{i}] skill[{j}] posterior=%A{p.LogOdds}"))

    let InferIsCorrect () =
        let posterior =
            engine.Infer<Bernoulli [] []>(isCorrect)

        printfn $"isCorrect posterior=%A{posterior}"
    

let Infer () =
    printfn $"Chapter2 start"
    //    CSharpVersion()
//
// 2.1
//    let input3 =
//        FileUtils.Load<Inputs>(DataPath, "Toy3")
//    NoisyAndModel.Construct input3 false true
//    NoisyAndModel.InferSkills()


//2.4 v1
//    let InputData =
//        FileUtils.Load<Inputs>(DataPath, "InputData")
//
//    NoisyAndModel.Construct InputData false true
//    NoisyAndModel.InferSkills()

//2.4 v2 check model use real skill data sample as input
//    SampleFromModel()

//2.6 random model, whole prior is 0.5
//    let InputData =
//        FileUtils.Load<Inputs>(DataPath, "InputData")
//
//    NoisyAndModel.Construct InputData false true
//    NoisyAndModel.InferSkills()
    
//2.6 learn with Beta for Guess Prob
//    let InputData =
//        FileUtils.Load<Inputs>(DataPath, "InputData")
//
//    LearnedNoisyAndModel.Construct InputData false true
//    LearnedNoisyAndModel.InferSkills()
    
//2.6 finally full observe with known skill
    let InputData =
        FileUtils.Load<Inputs>(DataPath, "InputData")


    NoisyAndModel.Construct InputData true true
    NoisyAndModel.InferSkills()
