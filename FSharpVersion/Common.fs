module FSharpVersion.Common

open Microsoft.ML.Probabilistic.Models

let engine = InferenceEngine()
engine.ShowFactorGraph <- true
