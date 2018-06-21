from dp import *

if __name__ == '__main__':
    print("building model...")
    model = distance_parser(vocab_size=args.vocab_size,
                            embed_size=args.embedsz,
                            hid_size=args.hidsz,
                            arc_size=len(ptb_parsed.arc_dictionary),
                            stag_size=len(ptb_parsed.stag_dictionary),
                            window_size=args.window_size,
                            dropout=args.dpout,
                            dropoute=args.dpoute,
                            dropoutr=args.dpoutr)
    if args.cuda:
        model.cuda()

    if os.path.isfile(parameter_filepath):
        print("Resuming from file: {}".format(parameter_filepath))
        checkpoint = torch.load(parameter_filepath)
        start_epoch = checkpoint['epoch']
        valid_precision = checkpoint['valid_precision']
        valid_recall = checkpoint['valid_recall']
        best_valid_f1 = checkpoint['valid_f1']
        model.load_state_dict(checkpoint['model_state_dict'])
        print("loaded model: epoch {}, valid_loss {}, "
              "valid_precision {}, valid_recall {}, valid_f1 {}".format(
            start_epoch, checkpoint['valid_loss'], valid_precision, \
            valid_recall, best_valid_f1))

    print("Evaluating valid... ")
    valid_loss, valid_arc_prec, valid_tag_prec, \
    valid_precision, valid_recall, valid_f1 = evaluate(model, ptb_parsed, 'valid')
    print("Evaluating test... ")
    test_loss, test_arc_prec, test_tag_prec, \
    test_precision, test_recall, test_f1= evaluate(model, ptb_parsed, 'test')
    print(valid_log_template.format(
        start_epoch,
        ' ', valid_loss, valid_arc_prec, valid_tag_prec,
        valid_precision, valid_recall, valid_f1,
        ' ', test_loss, test_arc_prec, test_tag_prec,
        test_precision, test_recall, test_f1))