import torch
from torch.autograd import Variable

def train(epoch, dataloader, net, criterion, optimizer, opt):
    net.train()
    for i, (sample_idx, annotation, adj_matrix, label_edge, label_attribute, label_lost, label_return) in enumerate(dataloader, 0):
        net.zero_grad()

        padding = torch.zeros(opt.batchSize, opt.n_node, opt.L, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 3)
        """
        軽いやつ
        adj_matrix      :(4, 2, 3, 888)         :(batch_size, 2[in, out], 3[value + row + col], max_nnz_am)
        annotation      :(4, 100, 3, 101)       :(batch_size, n_node, L, annotation_dim)
        init_input      :(4, 100, 3, 101)       :(batch_size, n_node, L, state_dim )
        label_edge      :(4, 3, 777)            :(batch_size, 3[value + row + col], max_nnz_label_edge)
        label_attribute :(4, 100, 101)          :(batch_size, n_node, label_attribute_dim)
        label_lost      :(4, 80)                :(batch_size, n_existing_node)
        label_return    :(4, 80)                :(batch_size, n_existing_node)
        
        本番
        adj_matrix      :(4, 2, 3, 6466561)     :(batch_size, 2[in, out], 3[value + row + col], max_nnz_am)
        annotation      :(4, 4359930, 3, 101)   :(batch_size, n_node, L, annotation_dim)
        init_input      :(4, 4359930, 3, 101)   :(batch_size, n_node, L, state_dim)
        label_edge      :(4, 3, 2067514)        :(batch_size, 3[value + row + col], max_nnz_label_edge)
        label_attribute :(4, 4359930, 101)      :(batch_size, n_node, label_attribute_dim)
        label_lost      :(4, 2118551)           :(batch_size, n_existing_node)
        label_return    :(4, 2118551)           :(batch_size, n_existing_node)
        """
        if opt.cuda:
            adj_matrix      = adj_matrix.cuda()
            annotation      = annotation.cuda()
            init_input      = init_input.cuda()
            label_edge      = label_edge.cuda()
            label_attribute = label_attribute.cuda()
            label_lost      = label_lost.cuda()
            label_return    = label_return.cuda()

        adj_matrix      = Variable(adj_matrix)
        annotation      = Variable(annotation)
        init_input      = Variable(init_input)
        label_edge      = Variable(label_edge)
        label_attribute = Variable(label_attribute)
        label_lost      = Variable(label_lost)
        label_return    = Variable(label_return)

        output = net(init_input)
        target = label_attribute

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.item()))
