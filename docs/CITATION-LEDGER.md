# Citation ledger

Every manuscript citation is linked to a verified primary record. “Direct”
means the cited paper reports the specific phenomenon stated; “methodological”
means it supports the method or distinction; “contextual” means it frames the
problem without being evidence for the present results.

| Key | Full reference | Claim supported | Type | Verification |
|---|---|---|---|---|
| [1] | Maynez, Narayan, Bohnet, and McDonald (2020), “On Faithfulness and Factuality in Abstractive Summarization,” ACL 2020, 1906–1919, doi:10.18653/v1/2020.acl-main.173. | Generated outputs can be unfaithful to an input document. | Direct/contextual | ACL Anthology record and DOI verified. |
| [2] | Zhou, Zhang, Poon, and Chen (2023), “Context-faithful Prompting for Large Language Models,” Findings of EMNLP 2023, 14544–14556, doi:10.18653/v1/2023.findings-emnlp.968. | LLMs can overlook contextual cues under knowledge conflict. | Direct/contextual | ACL Anthology record and DOI verified. |
| [3] | Xie, Zhang, Chen, Lou, and Su (2024), “Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts,” ICLR 2024, OpenReview: auKAUJZMO6. | Controlled conflict studies reveal prompt-dependent use of external evidence. | Direct/contextual | ICLR paper and OpenReview record verified. |
| [4] | Biderman et al. (2023), “Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling,” ICML 2023, PMLR 202:2397–2430. | Public model families and intermediate checkpoints enable training-dynamics analysis. | Direct/contextual | PMLR official record verified. |
| [5] | Groeneveld et al. (2024), “OLMo: Accelerating the Science of Language Models,” ACL 2024, 15789–15809, doi:10.18653/v1/2024.acl-long.841. | OLMo provides open weights, data, code, and intermediate checkpoints. | Direct/contextual | ACL Anthology record and DOI verified. |
| [6] | Power, Burda, Edwards, Babuschkin, and Misra (2022), “Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets,” arXiv:2201.02177. | Generalization can evolve non-monotonically and late in training in controlled tasks. | Direct but conceptual analogy | arXiv record verified; cited only for analogy, not as a language-model result. |
| [7] | Nanda, Chan, Lieberum, Smith, and Steinhardt (2023), “Progress Measures for Grokking via Mechanistic Interpretability,” ICLR 2023, arXiv:2301.05217. | Hidden progress measures can clarify apparent training transitions. | Direct but conceptual analogy | ICLR/arXiv record verified. |
| [8] | Elazar, Ravfogel, Jacovi, and Goldberg (2021), “Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals,” TACL 9:160–175, doi:10.1162/tacl_a_00359. | Probing/representation information should be separated from behavioral use. | Methodological | ACL Anthology record and DOI verified. |
| [9] | Burns, Ye, Klein, and Steinhardt (2023), “Discovering Latent Knowledge in Language Models Without Supervision,” ICLR 2023, arXiv:2212.03827. | Hidden states can contain knowledge that is not directly reflected in outputs. | Direct/contextual | ICLR/OpenReview and arXiv records verified. |
| [10] | Orgad, Toker, Gekhman, Reichart, Szpektor, Kotek, and Belinkov (2025), “LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations,” ICLR 2025, arXiv:2410.02707. | Internal truthfulness information can dissociate from generated behavior. | Direct/contextual | ICLR/OpenReview and arXiv records verified. |
| [11] | Feng, Russell, and Steinhardt (2025), “Monitoring Latent World States in Language Models with Propositional Probes,” ICLR 2025, arXiv:2406.19501. | Latent context/world-state information can be probed even when output behavior is unfaithful. | Direct/contextual | ICLR paper and arXiv record verified. |
| [12] | Vig, Gehrmann, Belinkov, Qian, Nevo, Singer, and Shieber (2020), “Investigating Gender Bias in Language Models Using Causal Mediation Analysis,” NeurIPS 33. | Internal interventions can estimate causal contributions to model outputs. | Methodological | Official NeurIPS record verified. |
| [13] | Meng, Bau, Andonian, and Belinkov (2022), “Locating and Editing Factual Associations in GPT,” NeurIPS 35, 17359–17372. | Causal tracing can localize internal contributions to factual predictions. | Methodological | Official NeurIPS paper verified. |
| [14] | Geiger, Lu, Icard, and Potts (2021), “Causal Abstractions of Neural Networks,” NeurIPS 34. | Interchange interventions provide a formal causal-abstraction perspective. | Methodological | Official NeurIPS record verified. |
| [15] | Zhang and Nanda (2024), “Towards Best Practices of Activation Patching in Language Models: Metrics and Methods,” ICLR 2024, arXiv:2309.16042. | Patching results depend on corruption, metric, and methodological choices. | Methodological | ICLR/arXiv record verified. |
| [16] | Team OLMo et al. (2025), “2 OLMo 2 Furious,” arXiv:2501.00656. | OLMo 2 model family and released intermediate checkpoints/artifacts. | Model-release context | arXiv record and DOI 10.48550/arXiv.2501.00656 verified. |
| [17] | Kim, Kim, Kwon, Yang, Jung, and Cha (2026), “How Training Data Shapes the Use of Parametric and In-Context Knowledge in Language Models,” ACL 2026, 23242–23257, doi:10.18653/v1/2026.acl-long.1064. | Recent checkpoint-wise analysis of parametric/in-context preference; motivates the narrower novelty claim. | Direct/contextual | ACL Anthology record and DOI verified. |
| [18] | Liu et al. (2024), “LLM360: Towards Fully Transparent Open-Source LLMs,” in Proceedings of the First Conference on Language Modeling (COLM). | Amber-7B release and intermediate training artifacts. | Model-release context | Official COLM accepted-paper listing/OpenReview record verified; arXiv:2312.06550 retained as the preprint identifier. |
| [19] | She, Li, Xing, Liu, and Ho (2025), “Linear Steerability in Language Models: When It Emerges and How It Evolves,” Findings of EMNLP 2025, 17821–17846, doi:10.18653/v1/2025.findings-emnlp.969. | Linear steerability can emerge at intermediate checkpoints and vary across concepts/families. | Direct/contextual | ACL Anthology record, pages, authors, and DOI verified. |
| [20] | Makelov, Lange, Geiger, and Nanda (2024), “Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching,” ICLR 2024, arXiv:2311.17030. | A patch can change output without uniquely identifying the intended feature or pathway. | Methodological | ICLR proceedings record and arXiv identifier verified. |
| [21] | Allen Institute for AI (2025), “OLMo-2-0425-1B,” Hugging Face model card. https://huggingface.co/allenai/OLMo-2-0425-1B. | Exact 1B released model identifier used in the trajectory. | Model metadata | Official model card and repository identifier verified; accessed 2026-08-31. |
| [22] | Allen Institute for AI (2024), “OLMo-2-1124-7B,” Hugging Face model card. https://huggingface.co/allenai/OLMo-2-1124-7B. | Exact 7B released model identifier used in the trajectory. | Model metadata | Official model-card citation metadata identifies the December 2024 release; repository identifier verified; accessed 2026-08-31. |

## Cross-check

All keys [1]–[22] occur in `MANUSCRIPT.md`, and every citation in that file
has one of these keys. No review article is used as evidence for a specific
original result when an original record is listed above. The two recent
Neurocomputing records used for journal fit are intentionally not included in
the manuscript bibliography because the paper does not rely on them for a
scientific claim.
