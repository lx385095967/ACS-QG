from typing import Dict, List, TextIO, Optional

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F
import math

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure

from nrl.metrics.threshold_metric import ThresholdMetric
from nrl.modules.span_rep_assembly import SpanRepAssembly
from nrl.common.span import Span
#######
# from bert.modeling import BertModel,BERTLayerNorm,BertConfig
from bert.bert_total import get_bert_total

########

@Model.register("span_detector")
class SpanDetector(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 stacked_encoder: Seq2SeqEncoder,
                 #######
                 config_path:None,
                 vocab_path:None,
                 model_path:None,
                 #########
                 predicate_feature_dim: int,
                 dim_hidden: int = 100,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(SpanDetector, self).__init__(vocab, regularizer)
        ##############
        _, _, model_bert = get_bert_total(config_path, vocab_path, model_path)
        self.bert = model_bert

        # self.bert = bert_load_state_dict(self.bert, torch.load("bert-base-uncased/pytorch_model.bin", map_location='cpu'))
        ###############
        self.dim_hidden = dim_hidden

        self.text_field_embedder = text_field_embedder
        self.predicate_feature_embedding = Embedding(2, predicate_feature_dim)#100

        self.embedding_dropout = Dropout(p=embedding_dropout)

        self.threshold_metric = ThresholdMetric()

        self.stacked_encoder = stacked_encoder

        self.span_hidden = SpanRepAssembly(self.stacked_encoder.get_output_dim(), self.stacked_encoder.get_output_dim(), self.dim_hidden)
        self.pred = TimeDistributed(Linear(self.dim_hidden, 1))

        # ############
        # self.pred_v = TimeDistributed(Linear(300, 1))
        # #####################
        # ##############
        # self.qa_outputs = TimeDistributed(Linear(300, 2))
        # #############


    def forward(self,  # type: ignore
                epoch,
                ####### 
                input_ids,
                input_mask,
                segment_ids,
                input_pred,
                token_to_orig_map,
                #######
                text: Dict[str, torch.LongTensor],
                #######
                # start_positions: torch.LongTensor,
                # end_positions: torch.LongTensor,
                ########
                predicate_indicator: torch.LongTensor,
                labeled_spans: torch.LongTensor = None,
                annotations: Dict = None,
                **kwargs):
        ##################

        
        all_encoder_layers, _ = self.bert(input_ids,segment_ids,input_mask)
        sequence_output = all_encoder_layers[-1]
        # embedded_text_input = all_encoder_layers[-1]
        embedded_text_input = bert2word2(sequence_output,token_to_orig_map,text['tokens'])#torch.Size([4, 19, 768])
        embedded_text_input = embedded_text_input.cuda()
        ####################


        if epoch<=0:
            alpha = 1
        else:
            alpha = 0.2
        # embedded_text_input = self.embedding_dropout(self.text_field_embedder(text))
        mask = get_text_field_mask(text)
        embedded_predicate_indicator = self.predicate_feature_embedding(predicate_indicator.long())
        # embedded_input_pred = self.predicate_feature_embedding(input_pred.long())
 
        # embedded_text_with_predicate_indicator = torch.cat([embedded_text_input, embedded_predicate_indicator], -1)
        # embedded_text_with_predicate_indicator = torch.cat([embedded_text_input, embedded_input_pred], -1)#[768+100]
        # batch_size, sequence_length, embedding_dim_with_predicate_feature = embedded_text_with_predicate_indicator.size()
        batch_size, sequence_length, embedding_dim_with_predicate_feature = embedded_text_input.size()
        # print(embedding_dim_with_predicate_feature)
        # import pdb;pdb.set_trace()

        # if self.stacked_encoder.get_input_dim() != embedding_dim_with_predicate_feature:
        #     raise ConfigurationError("The SRL model uses an indicator feature, which makes "
        #                              "the embedding dimension one larger than the value "
        #                              "specified. Therefore, the 'input_dim' of the stacked_encoder "
        #                              "must be equal to total_embedding_dim + 1.")
       
        # encoded_text = self.stacked_encoder(embedded_text_with_predicate_indicator, mask)
        encoded_text = self.stacked_encoder(embedded_text_input, mask)


        #################
        # encoded_text = bert2word2(encoded_text,token_to_orig_map,text['tokens'])
        # _,sequence_length,_=encoded_text.size()
        ####################

        span_hidden, span_mask = self.span_hidden(encoded_text, encoded_text, mask, mask)
        logits = self.pred(F.relu(span_hidden)).squeeze(2)

        #################谓词预测
        # logits_v = self.pred_v(encoded_text).squeeze(2)
        # loss_v = F.binary_cross_entropy_with_logits(logits_v, predicate_indicator.float(), weight=mask.float(), size_average=False)
        ###############
        ##############起止边界预测
        # logits_se = self.qa_outputs(encoded_text)
        # start_logits, end_logits = logits_se.split(1, dim=-1)
        # start_logits = start_logits.squeeze(-1)
        # end_logits = end_logits.squeeze(-1)
        # start_loss = distant_cross_entropy(start_logits, start_positions)
        # end_loss = distant_cross_entropy(end_logits, end_positions)
        # total_loss = (start_loss + end_loss) / 2
        #################
        
        probs = F.sigmoid(logits) * span_mask.float()
        output_dict = {"logits": logits, "probs": probs, 'span_mask': span_mask}

        if labeled_spans is not None:
            span_label_mask = (labeled_spans[:, :, 0] >= 0).squeeze(-1).long()
            prediction_mask = self.get_prediction_map(labeled_spans, span_label_mask, sequence_length, annotations=annotations)
            ###########FocalLoss
            fl = FocalLoss(logits=True)
            loss = fl(logits,prediction_mask,span_mask=span_mask)
            #############
            ############GHMLoss
            # ghmcloss = GHMC_Loss(bins=10, alpha=0.75)
            # loss = ghmcloss(probs,prediction_mask)
            ################
            ##############BCE_baseline

            # loss = F.binary_cross_entropy_with_logits(logits, prediction_mask, weight=span_mask.float(), size_average=False)
            # output_dict["loss"] = (1-alpha)*loss+alpha*(total_loss)
            ####################
            output_dict["loss"] = loss
            if not self.training:
                spans = self.to_scored_spans(probs, span_mask)
                self.threshold_metric(spans, annotations)

        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = mask
        return output_dict

    def to_scored_spans(self, probs, score_mask):
        probs = probs.data.cpu()
        score_mask = score_mask.data.cpu()
        batch_size, num_spans = probs.size()
        spans = []
        for b in range(batch_size):
            batch_spans = []
            for start, end, i in self.start_end_range(num_spans):
                if score_mask[b, i] == 1 and probs[b, i] > 0:
                    batch_spans.append((Span(start, end), probs[b, i]))
            spans.append(batch_spans)
        return spans

    def start_end_range(self, num_spans):
        n = int(.5 * (math.sqrt(8 * num_spans + 1) -1))

        result = []
        i = 0
        for start in range(n):
            for end in range(start, n):
                result.append((start, end, i))
                i += 1

        return result

    def get_prediction_map(self, spans, span_mask, seq_length, annotations=None):
        batchsize, num_spans, _ = spans.size()
        num_labels = int((seq_length * (seq_length+1))/2)
        labels = spans.data.new().resize_(batchsize, num_labels).zero_().float()
        spans = spans.data
        arg_indexes = (2 * spans[:,:,0] * seq_length - spans[:,:,0].float().pow(2).long() + spans[:,:,0]) / 2 + (spans[:,:,1] - spans[:,:,0])
        arg_indexes = arg_indexes * span_mask.data

        for b in range(batchsize):
            for s in range(num_spans):
                if span_mask.data[b, s] > 0:
                    labels[b, arg_indexes[b, s]] = 1

        return torch.autograd.Variable(labels)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor], remove_overlap=True) -> Dict[str, torch.Tensor]:
        probs = output_dict['probs']
        mask = output_dict['span_mask']
        spans = self.to_scored_spans(probs, mask)
        output_dict['spans'] = spans
        return output_dict
 

        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].data.cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        for predictions, length in zip(predictions_list, sequence_lengths):
            max_likelihood_sequence, _ = viterbi_decode(predictions[:length], transition_matrix)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_sequence]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = self.threshold_metric.get_metric(reset=reset)
        #if self.training:
            # This can be a lot of metrics, as there are 3 per class.
            # During training, we only really care about the overall
            # metrics, so we filter for them here.
            # TODO(Mark): This is fragile and should be replaced with some verbosity level in Trainer.
            #return {x: y for x, y in metric_dict.items() if "overall" in x}

        return metric_dict

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.

        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SpanDetector':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        stacked_encoder = Seq2SeqEncoder.from_params(params.pop("stacked_encoder"))
        predicate_feature_dim = params.pop("predicate_feature_dim")
        dim_hidden = params.pop("hidden_dim", 100)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        ######
        config_path = params.pop("config_path")
        vocab_path = params.pop("vocab_path")
        model_path = params.pop("model_path")

        ######
        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   stacked_encoder=stacked_encoder,
                   ######
                   config_path = config_path,
                   vocab_path = vocab_path,
                   model_path = model_path,
                   ###########
                   predicate_feature_dim=predicate_feature_dim,
                   dim_hidden = dim_hidden,
                   initializer=initializer,
                   regularizer=regularizer)

def perceptron_loss(logits, prediction_mask, score_mask):
    batch_size, seq_length, _ = logits.size()

    max_bad, _ = (logits + (prediction_mask == 0).float().log() + score_mask.float().log()).view(batch_size, -1).max(1)
    min_good, _ = (logits - (prediction_mask == 1).float().log()  - score_mask.float().log()).view(batch_size, -1).min(1) 

    bad_scores = (min_good.view(batch_size, 1, 1) - logits)
    bad_violations = bad_scores * (bad_scores < 0).float() * (prediction_mask == 0).float() * score_mask.float()
    #bad_norms = bad_violations.float().view(batch_size, -1).sum(1).view(batch_size, 1, 1).expand(batch_size, seq_length, seq_length)
    #bad_scores = bad_violations.masked_select(bad_violations < 0)
    #bad_scores = - bad_scores / bad_norms.masked_select(bad_violations < 0)
    #bad_violations = bad_scores

    good_scores = (max_bad.view(batch_size, 1, 1) - logits)
    good_violations = good_scores * (good_scores > 0).float() * prediction_mask.float() * score_mask.float()
    #good_norms = good_violations.float().view(batch_size, -1).sum(1).view(batch_size, 1, 1).expand(batch_size, seq_length, seq_length)
    #good_scores = good_violations.masked_select(good_violations > 0)
    #good_scores = good_scores / good_norms.masked_select(good_violations > 0)
    #good_violations = good_scores

    loss = good_violations.sum() - bad_violations.sum()

    return loss

def write_to_conll_eval_file(prediction_file: TextIO,
                             gold_file: TextIO,
                             verb_index: Optional[int],
                             sentence: List[str],
                             prediction: List[str],
                             gold_labels: List[str]):
    """
    Prints predicate argument predictions and gold labels for a single verbal
    predicate in a sentence to two provided file references.

    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    verb_index : Optional[int], required.
        The index of the verbal predicate in the sentence which
        the gold labels are the arguments for, or None if the sentence
        contains no verbal predicate.
    sentence : List[str], required.
        The word tokens.
    prediction : List[str], required.
        The predicted BIO labels.
    gold_labels : List[str], required.
        The gold BIO labels.
    """
    verb_only_sentence = ["-"] * len(sentence)
    if verb_index:
        verb_only_sentence[verb_index] = sentence[verb_index]

    conll_format_predictions = convert_bio_tags_to_conll_format(prediction)
    conll_format_gold_labels = convert_bio_tags_to_conll_format(gold_labels)

    for word, predicted, gold in zip(verb_only_sentence,
                                     conll_format_predictions,
                                     conll_format_gold_labels):
        prediction_file.write(word.ljust(15))
        prediction_file.write(predicted.rjust(15) + "\n")
        gold_file.write(word.ljust(15))
        gold_file.write(gold.rjust(15) + "\n")
    prediction_file.write("\n")
    gold_file.write("\n")


def convert_bio_tags_to_conll_format(labels: List[str]):
    """
    Converts BIO formatted SRL tags to the format required for evaluation with the
    official CONLL 2005 perl script. Spans are represented by bracketed labels,
    with the labels of words inside spans being the same as those outside spans.
    Beginning spans always have a opening bracket and a closing asterisk (e.g. "(ARG-1*" )
    and closing spans always have a closing bracket (e.g. "*)" ). This applies even for
    length 1 spans, (e.g "(ARG-0*)").

    A full example of the conversion performed:

    [B-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, O]
    [ "(ARG-1*", "*", "*", "*", "*)", "*"]

    Parameters
    ----------
    labels : List[str], required.
        A list of BIO tags to convert to the CONLL span based format.

    Returns
    -------
    A list of labels in the CONLL span based format.
    """
    sentence_length = len(labels)
    conll_labels = []
    for i, label in enumerate(labels):
        if label == "O":
            conll_labels.append("*")
            continue
        new_label = "*"
        # Are we at the beginning of a new span, at the first word in the sentence,
        # or is the label different from the previous one? If so, we are seeing a new label.
        if label[0] == "B" or i == 0 or label[1:] != labels[i - 1][1:]:
            new_label = "(" + label[2:] + new_label
        # Are we at the end of the sentence, is the next word a new span, or is the next
        # word not in a span? If so, we need to close the label span.
        if i == sentence_length - 1 or labels[i + 1][0] == "B" or label[1:] != labels[i + 1][1:]:
            new_label = new_label + ")"
        conll_labels.append(new_label)
    return conll_labels




def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss

#Focalloss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, re_duce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.re_duce = re_duce

    def forward(self, inputs, targets,span_mask,size_average = False):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight = span_mask.float(),reduce=True,size_average=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.re_duce:
            return torch.mean(F_loss)
        else:
            return F_loss
def bert_load_state_dict(model, state_dict):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()

    if metadata is not None:
        state_dict._metadata = metadata
    print(model)
    import pdb; pdb.set_trace()
    def load(module, prefix=''):
        print(prefix)
        import pdb; pdb.set_trace()
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    if len(missing_keys) > 0:
        # logger.info("Weights of {} not initialized from pretrained model: {}".format(
        #     model.__class__.__name__, missing_keys))
        print(missing_keys)
    if len(unexpected_keys) > 0:
        # logger.info("Weights from pretrained model not used in {}: {}".format(
        #     model.__class__.__name__, unexpected_keys))
        print(unexpected_keys)
    return model
# def bert2word(a,b,c):
#     bs,sl = c.size()
#     temp = torch.zeros(bs,sl,768)
#     for i,map in enumerate(b):
#         _n = 0
#         for index,item in enumerate(map):
#             if item == -1:
#                 _n = _n+1
#                 if _n !=2:
#                     continue
#                 else:
#                     break
#             else:
#                 if torch.mean(temp[i][int(item)])==0:
#                     temp[i][int(item)] = a[i][index]
#                 else:
#                     continue
#     return temp
def bert2word2(a,b,c):
    bs,sl = c.size()
    # bs,sl = c
    temp = torch.zeros(bs,sl,768)
    for i,map in enumerate(b):
        _n = 0
        dic_n = {}
        map = map[1:]
        for index,item in enumerate(map):
            if int(item) == -1:
                for k,v in dic_n.items():
                    temp[i][k] = temp[i][k]/v
                break
            else:
                if int(item) not in dic_n:
                    dic_n[int(item)] = 1
                else:
                    dic_n[int(item)] += 1            
            temp[i][int(item)] += a[i][index+1].cpu()
    return temp
def get_predspan(predspans,sequence_length):
    start_p = torch.zeros(len(predspans),sequence_length)
    end_p = torch.zeros(len(predspans),sequence_length)

    for b in range(len(predspans)):
        pred_spans = [s[0] for s in predspans[b] if s[1] >= 0.5]
        for item in pred_spans:
            start_p[b][item[0]]=1
            end_p[b][item[1]]=1
    
    return start_p,end_p

#######GHM LOSS#############
class GHM_Loss(torch.nn.Module):
    def __init__(self, bins, alpha):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx].cuda())


class GHMC_Loss(GHM_Loss):
    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


class GHMR_Loss(GHM_Loss):
    def __init__(self, bins, alpha, mu):
        super(GHMR_Loss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)
