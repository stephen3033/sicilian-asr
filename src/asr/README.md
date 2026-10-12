# ASR from IPA distributions

The language directories own lexical resources and linguistic policy. [Italian](ita/README.md) has a working prepared decoder; [Sicilian](scn/README.md) currently supports resource preparation and assessment.

[Shared WFST code](wfst/README.md) composes and searches graphs using the fixed IPA inventory. It does not import either language implementation. A future combined graph can use this shared layer without mixing today's language-specific resources.
