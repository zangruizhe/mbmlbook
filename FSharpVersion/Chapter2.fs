module FSharpVersion.Chapter2

open MBMLViews
open AssessingPeoplesSkills

[<Literal>]
let DataPath =
    __SOURCE_DIRECTORY__
    + "/../src/2. Assessing Peoples Skills/Data/"

let Infer() =
    let input3 = FileUtils.Load<Inputs>(DataPath, "Toy3")
    printfn $"%A{input3}"
