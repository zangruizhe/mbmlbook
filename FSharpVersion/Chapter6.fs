module FSharpVersion.Chapter6

open Microsoft.ML.Probabilistic.Learners
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Math
open Microsoft.ML.Probabilistic.Models.Attributes
open Microsoft.ML.Probabilistic.Collections

open System.Linq
open FSharpVersion.Common
open UnderstandingAsthma

[<Literal>]
let DataPath = __SOURCE_DIRECTORY__ + "/c6_data/"

let data =
    let allData = AllergenData()
    let tmp = allData.LoadDataFromTabDelimitedFile(DataPath)
    let list =  System.Collections.Generic.List<string>()
    list.Add("Mould")
    list.Add("Peanut")
    
    AllergenData.WithAllergensRemoved(allData, list)
    
    
type AsthmaModel()=
    let breakSymmetry = true
    let numYears, Years = GetRange "Years" 
    let numChildren, Children = GetRange "Children" 
    let numAllergens, Allergens = GetRange "Allergens" 
    let numClasses, Classes = GetRange "Classes"
    let sensitized = Get3DArrayVar<bool> "sensitized" Children Allergens Years
    let skinTest = Get3DArrayVar<bool> "skinTest" Children Allergens Years
    let igeTest = Get3DArrayVar<bool> "igeTest" Children Allergens Years
    let skinTestMissing = Get3DArrayVar<bool> "skinTestMissing" Children Allergens Years
    let igeTestMissing = Get3DArrayVar<bool> "igeTestMissing" Children Allergens Years
    
    let probSensClassPrior,probSensClass  = GetVarWithDist<Vector, Dirichlet> "probSensClass"
    do probSensClass.SetValueRange(Classes)
    
//    let probSensClass = Variable<Vector>.Random(probSensClassPrior).Named("probSensClass")
//    let probSensClass.SetValueRange(classes)
//    let sensClass = Variable.Array<int>(children).Named("sensClass")
//    let sensClass[children] = Variable.Discrete(probSensClass).ForEach(children)
//    let sensClassInitializer = Variable.New<IDistribution<int[]>>().Named("sensClassInitializer")
//    if (BreakSymmetry)
//    {
//        sensClass.InitialiseTo(sensClassInitializer);
//    }
