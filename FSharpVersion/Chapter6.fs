module FSharpVersion.Chapter6

open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Math

open FSharpVersion.Common
open UnderstandingAsthma

[<Literal>]
let DataPath =
    __SOURCE_DIRECTORY__
    + "/c6_data/SyntheticDataset.tsv"

let data =
    let allData = AllergenData()

    let tmp =
        allData.LoadDataFromTabDelimitedFile(DataPath)

    let list =
        System.Collections.Generic.List<string>()

    list.Add("Mould")
    list.Add("Peanut")

    AllergenData.WithAllergensRemoved(allData, list)

type ClinicalTrialModel() =
    let numControl, controlGroup =
        GetRange "controlGroup"

    let numtreat, treatedGroup =
        GetRange "treatedGroup"

    let recoveredControl =
        GetArrayVar "recoveredControl" controlGroup

    let recoveredTreated =
        GetArrayVar "recoveredTreated" treatedGroup

    let model =
        Variable.Bernoulli(0.5).Named("model")

    let probControl =
        Variable.Beta(1, 1).Named("probControl")

    let probTreated =
        Variable.Beta(1, 1).Named("probTreated")

    let probRecovery =
        Variable.Beta(1, 1).Named("probRecovery")

    do
        Variable.IfBlock
            (model)
            (fun _ ->
                recoveredControl[controlGroup] <- Variable
                    .Bernoulli(probControl)
                    .ForEach(controlGroup)

                recoveredTreated[treatedGroup] <- Variable
                    .Bernoulli(probTreated)
                    .ForEach(treatedGroup))
            (fun _ ->
                recoveredControl[controlGroup] <- Variable
                    .Bernoulli(probRecovery)
                    .ForEach(controlGroup)

                recoveredTreated[treatedGroup] <- Variable
                    .Bernoulli(probRecovery)
                    .ForEach(treatedGroup))

    member this.SetObservation (recoveredControlOb: bool []) (recoveredTreatedOb: bool []) =
        numControl.ObservedValue <- Array.length recoveredControlOb
        numtreat.ObservedValue <- Array.length recoveredTreatedOb

        recoveredControl.ObservedValue <- recoveredControlOb
        recoveredTreated.ObservedValue <- recoveredTreatedOb

    member this.Infer() =
        let model = engine.Infer<Bernoulli>(model)

        let probIfControl =
            engine.Infer<Beta>(probControl)

        let probIfTreated =
            engine.Infer<Beta>(probTreated)

        printfn $"post: model=%A{model}"
        printfn $"probIfControl: model=%A{probIfControl}"
        printfn $"probIfTreated: model=%A{probIfTreated}"

type AsthmaModel() =
    let numYears, Years = GetRange "Years"

    let numChildren, Children =
        GetRange "Children"

    let numAllergens, Allergens =
        GetRange "Allergens"

    let numClasses, Classes = GetRange "Classes"

    let sensitized =
        Get3DArrayVar<bool> "sensitized" Children Allergens Years

    let skinTest =
        Get3DArrayVar<bool> "skinTest" Children Allergens Years

    let igeTest =
        Get3DArrayVar<bool> "igeTest" Children Allergens Years

    let skinTestMissing =
        Get3DArrayVar<bool> "skinTestMissing" Children Allergens Years

    let igeTestMissing =
        Get3DArrayVar<bool> "igeTestMissing" Children Allergens Years

    let probSensClassPrior, probSensClass =
        GetVarWithDist<Vector, Dirichlet> "probSensClass"

    do probSensClass.SetValueRange(Classes)

    let sensClass =
        GetArrayVar "sensClass" Children

    do sensClass.[Children] <- Variable.Discrete(probSensClass).ForEach(Children)

    let sensClassInitializer =
        Variable
            .New<IDistribution<int []>>()
            .Named("sensClassInitializer")

    do
        sensClass.InitialiseTo(sensClassInitializer)
        |> ignore

    // Transition probabilities
    let probSens1Prior, probSens1 =
        Get2DVarWithDst<float, Beta> "probSens1" Allergens Classes

    let probGainPrior, probGain =
        Get3DVarWithDst<float, Beta> "probGain" Allergens Classes Years

    let probRetainPrior, probRetain =
        Get3DVarWithDst<float, Beta> "probRetain" Allergens Classes Years
    //    let x = probRetain[Years][Allergens,sensClass[Children]][Children][Allergens]

    // Emission probabilities
    let probSkinIfSensPrior, probSkinIfSens =
        GetVarWithDist<float, Beta> "probSkinIfSens"

    let probSkinIfNotSensPrior, probSkinIfNotSens =
        GetVarWithDist<float, Beta> "probSkinIfNotSens"

    let probIgeIfSensPrior, probIgeIfSens =
        GetVarWithDist<float, Beta> "probIgeIfSens"

    let probIgeIfNotSensPrior, probIgeIfNotSens =
        GetVarWithDist<float, Beta> "probIgeIfNotSens"

    do
        Variable.ForeachBlock Children (fun _ ->
            Variable.SwitchBlock sensClass[Children] (fun _ ->
                Variable.ForeachBlock Allergens (fun _ ->
                    ForeachBlockI Years (fun b ->
                        let year = b.Index
                        let yearIs0 = (year == 0).Named("year==0")
                        //                        let yearIsGr0 = (year >> 0).Named("year>0")

                        //                        let cur_class =
//                            sensClass.[Children].Named("cur_class")

                        Variable.IfBlock
                            yearIs0
                            (fun _ ->
                                sensitized.[year][Children, Allergens] <- Variable.Bernoulli(
                                    probSens1.[Allergens, sensClass.[Children]]
                                ))
                            (fun _ ->
                                let prevYear = (year - 1).Named("year - 1")

                                Variable.IfBlock
                                    (sensitized.[prevYear][Children, Allergens])
                                    (fun _ ->
                                        sensitized.[year][Children, Allergens] <- Variable.Bernoulli(
                                            probRetain.[year][Allergens, sensClass.[Children]]
                                        ))
                                    (fun _ ->
                                        sensitized.[year][Children, Allergens] <- Variable.Bernoulli(
                                            probGain.[year][Allergens, sensClass.[Children]]
                                        )))))))

    let SetTestPrior skinPrior igePrior =
        Variable.IfBlock (skinTestMissing.[Years][Children, Allergens]) (fun _ -> ()) (fun _ -> ())
        Variable.IfBlock (igeTestMissing.[Years][Children, Allergens]) (fun _ -> ()) (fun _ -> ())

    do
        Variable.ForeachBlock Children (fun _ ->
            Variable.ForeachBlock Allergens (fun _ ->
                Variable.ForeachBlock Years (fun _ ->
                    Variable.IfBlock
                        (sensitized.[Years][Children, Allergens])
                        (fun _ ->
                            Variable.IfBlock
                                (skinTestMissing.[Years][Children, Allergens])
                                (fun _ -> ())
                                (fun _ -> skinTest.[Years][Children, Allergens] <- Variable.Bernoulli(probSkinIfSens))

                            Variable.IfBlock
                                (igeTestMissing.[Years][Children, Allergens])
                                (fun _ -> ())
                                (fun _ -> igeTest.[Years][Children, Allergens] <- Variable.Bernoulli(probIgeIfSens)))
                        (fun _ ->
                            Variable.IfBlock
                                (skinTestMissing.[Years][Children, Allergens])
                                (fun _ -> ())
                                (fun _ ->
                                    skinTest.[Years][Children, Allergens] <- Variable.Bernoulli(probSkinIfNotSens))

                            Variable.IfBlock
                                (igeTestMissing.[Years][Children, Allergens])
                                (fun _ -> ())
                                (fun _ -> igeTest.[Years][Children, Allergens] <- Variable.Bernoulli(probIgeIfNotSens))))))

    member this.SetPriors
        (data: AllergenData)
        (numVulnerabilities: int)
        (beliefs: UnderstandingAsthma.AsthmaModel.Beliefs)
        =
        let nY = AllergenData.NumYears
        let nN = data.DataCountChild.Length
        let nA = data.NumAllergens
        let useUniformClassPrior = true

        match beliefs = null with
        | true ->
            probSensClassPrior.ObservedValue <-
                match useUniformClassPrior with
                | true -> Dirichlet.PointMass(Vector.Constant(numVulnerabilities, 1.0 / float numVulnerabilities))
                | false -> Dirichlet.Symmetric(numVulnerabilities, 0.1)

            probSens1Prior.ObservedValue <- Array2D.init nA numVulnerabilities (fun _ _ -> Beta(1, 1))

            probGainPrior.ObservedValue <-
                Array.init nY (fun _ -> Array2D.init nA numVulnerabilities (fun _ _ -> Beta(1, 1)))

            probRetainPrior.ObservedValue <-
                Array.init nY (fun _ -> Array2D.init nA numVulnerabilities (fun _ _ -> Beta(1, 1)))

            probSkinIfSensPrior.ObservedValue <- Beta(2.0, 1)
            probSkinIfNotSensPrior.ObservedValue <- Beta(1, 2.0)
            probIgeIfSensPrior.ObservedValue <- Beta(2.0, 1)
            probIgeIfNotSensPrior.ObservedValue <- Beta(1, 2.0)
        | false ->
            probSensClassPrior.ObservedValue <- beliefs.ProbVulnerabilityClass

            probSens1Prior.ObservedValue <-
                Array2D.init nA numVulnerabilities (fun a v -> beliefs.ProbSensitizationAgeOne.[a, v])

            probGainPrior.ObservedValue <-
                Array.init nY (fun y ->
                    Array2D.init nA numVulnerabilities (fun a v -> beliefs.ProbGainSensitization.[y][a, v]))

            probRetainPrior.ObservedValue <-
                Array.init nY (fun y ->
                    Array2D.init nA numVulnerabilities (fun a v -> beliefs.ProbRetainSensitization.[y][a, v]))

            probSkinIfSensPrior.ObservedValue <- beliefs.ProbSkinIfSensitized
            probSkinIfNotSensPrior.ObservedValue <- beliefs.ProbSkinIfNotSensitized
            probIgeIfSensPrior.ObservedValue <- beliefs.ProbIgEIfSensitized
            probIgeIfNotSensPrior.ObservedValue <- beliefs.ProbIgEIfNotSensitized

    member this.InitializeMessages (data: AllergenData) (numVulnerabilities: int) =
        let nN = data.DataCountChild.Length

        let discreteUniform =
            Discrete.Uniform(numVulnerabilities)

        sensClassInitializer.ObservedValue <-
            Distribution<int>.Array
                (Array.init nN (fun _ -> Discrete.PointMass(discreteUniform.Sample(), numVulnerabilities)))

    member this.SetObservation (data: AllergenData) (numVulnerabilities: int) =
        let nY = AllergenData.NumYears
        let nN = data.DataCountChild.Length
        let nA = data.NumAllergens

        // Observations
        numYears.ObservedValue <- nY
        numChildren.ObservedValue <- nN
        numAllergens.ObservedValue <- nA
        numClasses.ObservedValue <- numVulnerabilities

        skinTest.ObservedValue <-
            Array.init nY (fun y ->
                Array2D.init nN nA (fun n a ->
                    let tmp = (data.SkinTestData.[y][n])[a]
                    (tmp.HasValue = true && tmp.Value = 1)))

        igeTest.ObservedValue <-
            Array.init nY (fun y ->
                Array2D.init nN nA (fun n a ->
                    let tmp = (data.IgeTestData.[y][n])[a]
                    tmp.HasValue = true && tmp.Value = 1))

        skinTestMissing.ObservedValue <-
            Array.init nY (fun y -> Array2D.init nN nA (fun n a -> ((data.SkinTestData.[y][n])[a]).HasValue = false))

        igeTestMissing.ObservedValue <-
            Array.init nY (fun y -> Array2D.init nN nA (fun n a -> ((data.IgeTestData.[y][n])[a]).HasValue = false))

    member this.infer() =
        let Sensitization =
            engine.Infer<Bernoulli [,] []>(sensitized)

        let ProbSkinIfSensitized =
            engine.Infer<Beta>(probSkinIfSens)

        let ProbSkinIfNotSensitized =
            engine.Infer<Beta>(probSkinIfNotSens)

        let ProbIgEIfSensitized =
            engine.Infer<Beta>(probIgeIfSens)

        let ProbIgEIfNotSensitized =
            engine.Infer<Beta>(probIgeIfNotSens)

        let ProbSensitizationAgeOne =
            engine.Infer<Beta [,]>(probSens1)

        let ProbGainSensitization =
            engine.Infer<Beta [,] []>(probGain)

        let ProbRetainSensitization =
            engine.Infer<Beta [,] []>(probRetain)

        let VulnerabilityClass =
            engine.Infer<Discrete []>(sensClass)

        printfn $"post: Sensitization=%A{Sensitization}"
        printfn $"post: ProbSkinIfSensitized=%A{ProbSkinIfSensitized}"
        printfn $"post: ProbSkinIfNotSensitized=%A{ProbSkinIfNotSensitized}"
        printfn $"post: ProbIgEIfSensitized=%A{ProbIgEIfSensitized}"
        printfn $"post: ProbIgEIfNotSensitized=%A{ProbIgEIfNotSensitized}"
        printfn $"post: ProbSensitizationAgeOne=%A{ProbSensitizationAgeOne}"
        printfn $"post: ProbGainSensitization=%A{ProbGainSensitization}"
        printfn $"post: ProbRetainSensitization=%A{ProbRetainSensitization}"
        printfn $"post: VulnerabilityClass=%A{VulnerabilityClass}"

let private ClinicalInfer () =
    [| 20; 60; 100 |]
    |> Array.iter (fun num ->
        let clinical = ClinicalTrialModel()

        let recoveredControl =
            Program.GenerateMockClinicalTrialData(num, 0.4)

        let recoveredTreated =
            Program.GenerateMockClinicalTrialData(num, 0.65)

        clinical.SetObservation recoveredControl recoveredTreated
        clinical.Infer())

let private AsthmaInfer () =
    let numVulnerabilities = 5
    let model = AsthmaModel()
    engine.NumberOfIterations <- 30
    model.InitializeMessages data numVulnerabilities
    model.SetObservation data numVulnerabilities
    model.SetPriors data numVulnerabilities null
    model.infer ()

let Infer () =
    //    ClinicalInfer()
    AsthmaInfer()
