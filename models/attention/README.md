## Attention Modules

### Self Attention

Self Attention Works by seeing how similar each word is to all of the words in the sentence, including itself.

The attention component of a network manages and quantifies the interdependence.
- General Attention: Between Input and Output elements.
- Self Attention: Within input elements.

- Self Attention (Cheng 2016): Relating different positions of the same input sequence.Theoretically the self attention
can adopt any score functions above, but just replace the target sequence with the same input sequence.

- Global/Soft (Xu 2015): Attending to the entire input state space

- Local/Hard (Xu 2015, Luong 2015): Attending to the part of input state space, ie: a patch of the input image.

*Equation/Math:*

- Query: The object we are using to search
- Keys: indexes which are used to calculate the similarities between the query and keys
- Values: What we get as a result of the search

Attention(Q,K,V) = Softmax(QK^T/sqrt(dk))*V

Sentence * Linear() -> Query Vector
Sentence * Linear() -> Key Vector
Sentence * Linear() -> Value Vector

`Q*KT -> Softmax(Q*KT/sqrt(dk))` Scaled Dot Product Similarities Between Query and Keys

Then `Multiplied with the Value Vector, which tells us how much influence each word should have on the final encoding for any given word.`

### Masked Self Attention

In self attention, we use similarities between the words and itself and everything which comes after it.
Whereas, in masked self attention we will only use the similarity between the words and itself and ignore everything that came after it.
They can never look ahead at what comes next.

### Multiheaded Attention

If we apply the same self attention multiple times in a parallel fashion, with different weight matrices.

Then concatenate the output from all attention heads and multiply with a weight matrix W0 that was trained jointly with the model, we get an output that captures information from all attention heads.

![Image](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

Credit: [Blog by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)