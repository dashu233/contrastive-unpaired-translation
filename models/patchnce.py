from packaging import version
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, weighted = None):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        if weighted is not None:
            if self.opt.prob_weighted:
                l_neg_curbatch.masked_fill_(diagonal, -10.0)
                l_neg = l_neg_curbatch.view(-1, npatches)
                #
                # print('weight:', weighted.shape)
                # print('lpos:', l_pos.shape)
                # print('lneg:', l_neg.shape)

                weighted = weighted.view(-1,npatches)
                weighted = torch.cat((torch.ones([l_pos.shape[0],1],device=l_pos.device),weighted),dim=1)

                out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

                out = out.exp()
                out = out*weighted
                out = out/torch.sum(out,dim=1,keepdim=True)
                loss = -torch.log(out[:,0])
                return loss
            else:
                l_neg_curbatch = l_neg_curbatch*weighted
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                            device=feat_q.device))

        return loss

class PatchNCELoss_maxcut(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, sample_id):
        '''
        feat_q: b, patch, c
        feat_k: b, h x w, c
        sample_id: patch
        '''
        dev = feat_q.device
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[-1]
        feat_k = feat_k.detach()

        # pos logit: maximum cosine similarity
        sim_qk = torch.bmm(feat_q, feat_k.transpose(2, 1)) # b x num_patch x (h x w)
        l_pos, pos_idx = sim_qk.max(dim=-1) # b, num_patch

        # neg logit
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k_sample = feat_k[:, sample_id, :]
        feat_k_sample = feat_k_sample.view(batch_dim_for_bmm, -1, dim)

        npatches = feat_q.size(1)
        l_curbatch = torch.bmm(feat_q, feat_k_sample.transpose(2, 1))

        diagonal = torch.eye(npatches, device=dev, dtype=self.mask_dtype)[None, :, :]
        idx = torch.arange(npatches, dtype=torch.long, device=dev)
        l_max = l_curbatch.clone()
        l_max[:, idx, idx] = l_pos
        l_max = l_max.view(-1, npatches) # nce loss with max cosine similarity as positives
        l_same = l_curbatch.view(-1, npatches) # nce loss with embeddings from the same location as positives

        labels = torch.tensor(list(range(npatches)) * batchSize, dtype=torch.long, device=dev)
        loss_max = self.cross_entropy_loss(l_max / self.opt.nce_T, labels)
        loss_same = self.cross_entropy_loss(l_same / self.opt.nce_T, labels)

        loss = 0.5 * loss_same + 0.5 * loss_max
        # loss = 1.0 * loss_same + 0.0 * loss_max
        return loss

class PatchNCELoss_bicut(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.similarity_function = self._get_similarity_function()
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def _get_similarity_function(self):

        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        return self._cosine_simililarity

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, M, C)
        # v shape: (N, M)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        feat_k = feat_k.detach()
        l_pos = self.cos(feat_q,feat_k)
        l_pos = l_pos.view(batchSize, 1)
        l_neg_curbatch = self.similarity_function(feat_q.view(batchSize,1,-1),feat_k.view(1,batchSize,-1))
        l_neg_curbatch = l_neg_curbatch.view(1,batchSize,-1)
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(batchSize, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, batchSize)
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss