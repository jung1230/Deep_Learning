    class BasicEncoderXXXXX(nn.Module):
        """
        The BasicEncoder in TransformerFG consists of a layer of self-attention (SA) followed
        by a purely feed-forward layer (FFN).  The job of the SA layer is for the network to
        figure out what parts of an input sentence are relevant to what other parts of the
        same sentence in the process of learning how to translate a source-language sentence into
        a target-language sentence. The output of SA goes through FFN and the output of FFN becomes
        the output of the BasicEncoder.  To mitigate the problem of vanishing gradients in the FG
        transformer design, the output of each of the two components --- SA and FFN --- is subject
        to LayerNorm and a residual connection used that wraps around both the component and the
        LayerNorm which follows. Deploying a stack of BasicEncoder instances becomes easier if the 
        output tensor from a BasicEncoder has the same shape as its input tensor.  

        The SelfAttention layer mentioned above consists of a number of AttentionHead instances, 
        with each AttentionHead making an independent assessment of what to say about the 
        inter-relationships between the different parts of an input sequence. It is the embedding
        axis that is segmented out into disjoint slices for each AttentionHead instance.The 
        calling SelfAttention layer concatenates the outputs from all its AttentionHead instances 
        and presents the concatenated tensor as its own output.

        ClassPath:  TransformerFG  ->   BasicEncoder
        """
        def __init__(self, dls, xformer, num_atten_heads):
            super(TransformerFG.BasicEncoder, self).__init__()
            self.dls = dls
            self.embedding_size = xformer.embedding_size                                             
            self.max_seq_length = xformer.max_seq_length                                                     
            self.num_atten_heads = num_atten_heads
            self.self_attention_layer = xformer.SelfAttention(dls, xformer, num_atten_heads)                           ## (A)
            self.norm1 = nn.LayerNorm(self.embedding_size)                                                             ## (B)
            ## What follows are the linear layers for the FFN (Feed Forward Network) part of a BasicEncoder
#            self.W1 =  nn.Linear( self.max_seq_length * self.embedding_size, self.max_seq_length * 2 * self.embedding_size )
            self.W1 =  nn.Linear( self.embedding_size, 2 * self.embedding_size )
#            self.W2 =  nn.Linear( self.max_seq_length * 2 * self.embedding_size, self.max_seq_length * self.embedding_size ) 
            self.W2 =  nn.Linear( 2 * self.embedding_size, self.embedding_size ) 
            self.norm2 = nn.LayerNorm(self.embedding_size)                                                             ## (C)

        def forward(self, sentence_tensor):
            sentence_tensor = sentence_tensor.float()
            self_atten_out = self.self_attention_layer(sentence_tensor).to(self.dls.device)                            ## (D)
            normed_atten_out = self.norm1(self_atten_out + sentence_tensor)                                            ## (E)               

#            basic_encoder_out =  nn.ReLU()(self.W1( normed_atten_out.view(sentence_tensor.shape[0],-1) ))              ## (F)
            basic_encoder_out =  nn.ReLU()(self.W1( normed_atten_out ))              ## (F)

            basic_encoder_out =  self.W2( basic_encoder_out )                                                          ## (G)

#            basic_encoder_out = basic_encoder_out.view(sentence_tensor.shape[0], self.max_seq_length, self.embedding_size ) 
            ## for the residual connection and layer norm for FC layer:
            basic_encoder_out =  self.norm2(basic_encoder_out  + normed_atten_out)                                     ## (H)
            return basic_encoder_out

---------------------------------------------------------------------------------------------------------------------



    class BasicDecoderWithMaskingXXXX(nn.Module):
        """
        As with the basic encoder, while a basic decoder also consists of a layer of SelfAttention 
        followed by a Feedforward Network (FFN) layer, but now there is a layer of CrossAttention 
        interposed between the two.  The output from each of these three components of a basic 
        decoder passes through a LayerNorm layer. Additionally, you have a residual connection from 
        the input at each component to the output from the LayerNorm layer.

        An important feature of the BasicDecoder is the masking of the target sentences during the
        training phase in order to ensure that each predicted word in the target language depends only 
        on the target words that have been seen PRIOR to that point. This recursive backward dependency 
        is referred to as "autoregressive masking". In the implementation shown below, the masking is 
        initiated and its updates established by MasterDecoderWithMasking.

        ClassPath:  TransformerFG  ->   BasicDecoderWithMasking
        """
        def __init__(self, dls, xformer, num_atten_heads, masking=True):
            super(TransformerFG.BasicDecoderWithMasking, self).__init__()
            self.dls = dls
            self.masking = masking
            self.embedding_size = xformer.embedding_size                                             
            self.max_seq_length = xformer.max_seq_length                                                     
            self.num_atten_heads = num_atten_heads
            self.qkv_size = self.embedding_size // num_atten_heads
            self.self_attention_layer = xformer.SelfAttention(dls, xformer, num_atten_heads)
            self.norm1 = nn.LayerNorm(self.embedding_size)
            self.cross_attn_layer = xformer.CrossAttention(dls, xformer, num_atten_heads)
            self.norm2 = nn.LayerNorm(self.embedding_size)
            ## What follows are the linear layers for the FFN (Feed Forward Network) part of a BasicDecoder
#            self.W1 =  nn.Linear( self.max_seq_length * self.embedding_size, self.max_seq_length * 2 * self.embedding_size )
            self.W1 =  nn.Linear( self.embedding_size, 2 * self.embedding_size )
#            self.W2 =  nn.Linear( self.max_seq_length * 2 * self.embedding_size, self.max_seq_length * self.embedding_size ) 
            self.W2 =  nn.Linear( 2 * self.embedding_size, self.embedding_size ) 
            self.norm3 = nn.LayerNorm(self.embedding_size)

        def forward(self, sentence_tensor, final_encoder_out, mask):   
            masked_sentence_tensor = sentence_tensor
            if self.masking:
                masked_sentence_tensor = self.apply_mask(sentence_tensor, mask)
            ## for self attention
            Z_concatenated = self.self_attention_layer(masked_sentence_tensor).to(self.dls.device)
            Z_out = self.norm1(Z_concatenated + masked_sentence_tensor)                     
            ## for cross attention
            Z_out2  = self.cross_attn_layer( Z_out, final_encoder_out).to(self.dls.device)
            Z_out2 = self.norm2( Z_out2 )
            ## for FFN:
#            basic_decoder_out =  nn.ReLU()(self.W1( Z_out2.view(sentence_tensor.shape[0],-1) ))                  
            basic_decoder_out =  nn.ReLU()(self.W1( Z_out2 ))              ## (F)
            basic_decoder_out =  self.W2( basic_decoder_out )                                                    
#            basic_decoder_out = basic_decoder_out.view(sentence_tensor.shape[0], self.max_seq_length, self.embedding_size )
            basic_decoder_out =  basic_decoder_out  + Z_out2 
            basic_decoder_out = self.norm3( basic_decoder_out )
            return basic_decoder_out
