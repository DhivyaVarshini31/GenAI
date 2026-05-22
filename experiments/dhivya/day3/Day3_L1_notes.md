# What is prompt chaining

[Prompt chaining is a method where the output of one AI prompt is used as the input for another prompt. It helps break a large task into smaller steps and makes the output more accurate and organized.]

## When is chaining better than one big prompt

- [When the task has multiple steps.]
- [When it is easier to debug step by step]
- [When outputs need better accuracy and structure.]

## My chain idea for today

- Call 1 input: [Recipe ingredients text]
- Call 1 output: [ Extracted ingredient names and quantities]
- Call 2 input: [Extracted ingredient list from Call 1]
- Call 2 output: [Categorized shopping list]

## Questions for Dev

1. [How can I add better error handling to the chain?]
2. [How can I improve the accuracy of ingredient extraction?]

