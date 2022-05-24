// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace AssessingPeoplesSkills.Models
{
    /// <summary>
    ///     The model unrolled.
    /// </summary>
    public class UnrolledModel : NoisyAndModel
    {
        /// <summary>
        ///     The message histories.
        /// </summary>
        internal readonly Dictionary<string, List<Bernoulli>> MessageHistories =
            new Dictionary<string, List<Bernoulli>>();

        /// <summary>
        ///     The is correct variable.
        /// </summary>
        private Variable<bool>[][] isCorrect;

        /// <summary>
        ///     The unrolled skills.
        /// </summary>
        private Variable<bool>[][] unrolledSkills;

        /// <summary>
        ///     Gets or sets a value indicating whether to use exact inference.
        /// </summary>
        internal bool ExactInference { get; set; }

        /// <summary>
        ///     The set observed values.
        /// </summary>
        /// <param name="inputData">The input data.</param>
        /// <param name="parameters">The parameters.</param>
        /// <exception cref="System.ArgumentException">parameters should be length 2</exception>
        /// <exception cref="System.NotSupportedException">Inference not supported for this model configuration</exception>
        /// <exception cref="NotSupportedException"></exception>
        public override void SetObservedValues(Inputs inputData, params object[] parameters)
        {
            if (parameters == null || parameters.Length != 2)
                throw new ArgumentException("parameters should be length 2");

            var observeSkillTrue = (bool) parameters[0];
            var observeIsCorrect = (bool) parameters[1];

            if (observeSkillTrue || !observeIsCorrect)
                throw new NotSupportedException("Inference not supported for this model configuration");

            SkillNames = inputData.Quiz.SkillNames;

            NumberOfPeople = inputData.NumberOfPeople;
            NumberOfSkills = inputData.Quiz.NumberOfSkills;
            NumberOfQuestions = inputData.Quiz.NumberOfQuestions;

            NumberOfSkillsForEachQuestion = inputData.Quiz.NumberSkillsForQuestion;

            SkillsNeeded = inputData.Quiz.SkillsForQuestion;

            FinishModelConstruction();

            for (var p = 0; p < NumberOfPeople; p++)
            for (var q = 0; q < NumberOfQuestions; q++)
                isCorrect[p][q].ObservedValue = inputData.IsCorrect[p][q];
        }

        /// <summary>
        ///     Does the inference.
        /// </summary>
        /// <param name="results">
        ///     The results.
        /// </param>
        public override void DoInference(ref Results results)
        {
            InferSkills(ref results);
        }

        /// <summary>
        ///     Finishes the model construction.
        /// </summary>
        private void FinishModelConstruction()
        {
            unrolledSkills = new Variable<bool>[NumberOfPeople][];
            isCorrect = new Variable<bool>[NumberOfPeople][];

            for (var p = 0; p < NumberOfPeople; p++)
            {
                var personSuffix = NumberOfPeople == 1 ? string.Empty : string.Format("-{0}", p);

                // Skills for person
                unrolledSkills[p] = new Variable<bool>[NumberOfSkills];
                for (var s = 0; s < NumberOfSkills; s++)
                {
                    unrolledSkills[p][s] =
                        Variable.Bernoulli(ProbabilityOfSkillTrue).Named(SkillNames[s] + personSuffix);
                    unrolledSkills[p][s].AddAttribute(new DivideMessages(false));
                    unrolledSkills[p][s].AddAttribute(new ListenToMessages {Containing = "_uses"});
                }

                // Answers for person
                isCorrect[p] = new Variable<bool>[NumberOfQuestions];
                for (var q = 0; q < NumberOfQuestions; q++)
                    isCorrect[p][q] = Variable.New<bool>().Named("isCorrect" + q + personSuffix);

                if (!ExactInference)
                {
                    QuestionModelPart(p);
                }
                else
                {
                    // Cut the loop on the first skill variable (cutset conditioning)
                    var oldSkill = unrolledSkills[p][0];
                    unrolledSkills[p][0] = Variable.Constant(true);
                    using (Variable.If(oldSkill))
                    {
                        QuestionModelPart(p, "_T");
                    }

                    unrolledSkills[p][0] = Variable.Constant(false);
                    using (Variable.IfNot(oldSkill))
                    {
                        QuestionModelPart(p, "_F");
                    }

                    unrolledSkills[p][0] = oldSkill;
                }
            }
        }

        /// <summary>
        ///     Question part of the model
        /// </summary>
        /// <param name="person">The person.</param>
        /// <param name="suffix">The suffix.</param>
        /// <exception cref="System.Exception">Unrolling not implemented if more than two skills needed for a question</exception>
        private void QuestionModelPart(int person, string suffix = "")
        {
            for (var question = 0; question < NumberOfQuestions; question++)
            {
                var skillsNeeded = SkillsNeeded[question];
                switch (skillsNeeded.Length)
                {
                    case 1:
                        isCorrect[person][question].SetTo(
                            Factors.AddNoise(
                                unrolledSkills[person][skillsNeeded[0]],
                                ProbabilityOfNotMistake,
                                ProbabilityOfGuess));
                        break;
                    case 2:
                        var hasSkills =
                            (unrolledSkills[person][skillsNeeded[0]] & unrolledSkills[person][skillsNeeded[1]])
                            .Named("hasSkills" + (question + 1) + suffix);

                        isCorrect[person][question].SetTo(
                            Factors.AddNoise(hasSkills, ProbabilityOfNotMistake, ProbabilityOfGuess));
                        break;
                    default:
                        throw new Exception("Unrolling not implemented if more than two skills needed for a question");
                }
            }
        }

        /// <summary>
        ///     Infers the skills.
        /// </summary>
        /// <param name="results">
        ///     The results.
        /// </param>
        private void InferSkills(ref Results results)
        {
            // Engine.BrowserMode = BrowserMode.Always;
            // Engine.ShowFactorGraph = true;
            Engine.NumberOfIterations = 5;
            Engine.MessageUpdated += EngineMessageUpdated;

            Engine.Compiler.GivePriorityTo(typeof(ReplicateOp_NoDivide));
            results.SkillsPosteriors = new Bernoulli[NumberOfPeople][];
            for (var p = 0; p < NumberOfPeople; p++)
            {
                results.SkillsPosteriors[p] = new Bernoulli[NumberOfSkills];
                for (var j = 0; j < NumberOfSkills; j++)
                    results.SkillsPosteriors[p][j] = Engine.Infer<Bernoulli>(unrolledSkills[p][j]);
            }
        }

        /// <summary>
        ///     Handles the engine message updated.
        /// </summary>
        /// <param name="algorithm">The algorithm.</param>
        /// <param name="messageEvent">The <see cref="MessageUpdatedEventArgs" /> instance containing the event data.</param>
        private void EngineMessageUpdated(IGeneratedAlgorithm algorithm, MessageUpdatedEventArgs messageEvent)
        {
            if (!MessageHistories.ContainsKey(messageEvent.MessageId))
                MessageHistories[messageEvent.MessageId] = new List<Bernoulli>();

            // Console.WriteLine(messageEvent);
            if (messageEvent.Message is Bernoulli)
                MessageHistories[messageEvent.MessageId].Add((Bernoulli) messageEvent.Message);
        }
    }
}
