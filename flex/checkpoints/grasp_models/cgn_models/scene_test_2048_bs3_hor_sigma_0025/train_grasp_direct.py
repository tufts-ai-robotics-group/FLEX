import os
import sys
import argparse
import math
from datetime import datetime
import numpy as np
import socket
import importlib
import time
from tqdm import tqdm
import glob
import json
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(ROOT_DIR))

try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    TF2 = True
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    import tensorflow as tf
    TF2 = False

POINT_DIR_NGC = os.path.join(ROOT_DIR, 'pointnet2')
if os.path.exists(POINT_DIR_NGC):
    sys.path.append(os.path.join(POINT_DIR_NGC, 'models'))
    sys.path.append(os.path.join(POINT_DIR_NGC, 'utils'))
else:
    sys.path.append(os.path.join(BASE_DIR, 'pointnet2',  'models'))
    sys.path.append(os.path.join(BASE_DIR, 'pointnet2',  'utils'))
    
import provider
import sample
import utilities
from data import PointCloudReader, load_mesh_path_and_scale, preprocess_pc_for_inference, load_filtered_contact_data, load_scene_contacts, load_obj_scales_cats, inverse_transform, center_pc_convert_cam
from summaries import top_grasp_acc_summaries, build_summary_ops, build_file_writers
from tf_train_ops import load_labels_and_losses, build_train_op
from surface_grasp_estimator import GraspEstimator


def train(global_config, LOG_DIR):

    if 'train_on_scenes' in global_config['DATA'] and global_config['DATA']['train_on_scenes']:
        mesh_scales, mesh_cats = load_obj_scales_cats(global_config['DATA']['data_path'])
        contact_infos, scene_obj_paths, scene_obj_transforms = load_scene_contacts(global_config['DATA']['data_path'])
        num_train_samples = len(contact_infos)-global_config['DATA']['num_test_scenes']
        num_test_samples = global_config['DATA']['num_test_scenes']
    else:
        scene_obj_paths, scene_obj_transforms = None, None
        train_contact_paths, test_contact_paths, contact_infos, mesh_scales = load_filtered_contact_data(global_config['DATA']['data_path'], min_pos_contacts=1, classes=global_config['DATA']['classes'])
        num_train_samples = len(train_contact_paths)
        num_test_samples = len(test_contact_paths)
        
    print('using %s meshes' % (num_train_samples + num_test_samples))
    
    if 'train_and_test' in global_config['DATA'] and global_config['DATA']['train_and_test']:
        num_train_samples = num_train_samples + num_test_samples
        num_test_samples = 0
        print('using train and test data')

    pcreader = PointCloudReader(
        root_folder=global_config['DATA']['data_path'],
        batch_size=global_config['OPTIMIZER']['batch_size'],
        num_grasp_clusters=None,
        estimate_normals=global_config['DATA']['input_normals'],
        npoints=global_config['DATA']['num_point'],
        raw_num_points=global_config['DATA']['raw_num_points'],
        use_uniform_quaternions = global_config['DATA']['use_uniform_quaternions'],
        run_in_another_process = False,
        mesh_scales = mesh_scales,
        scene_obj_paths = scene_obj_paths,
        scene_obj_transforms = scene_obj_transforms,
        num_train_samples = num_train_samples,
        num_test_samples = num_test_samples,
        use_farthest_point = global_config['DATA']['use_farthest_point'],
        intrinsics=global_config['DATA']['intrinsics']
    )

    with tf.Graph().as_default():
        
        # Build the model
        grasp_estimator = GraspEstimator(global_config)

        ops = grasp_estimator.build_network()
        
        # contact_tensors = load_contact_grasps(contact_infos, global_config['DATA'])
        
        loss_ops = load_labels_and_losses(grasp_estimator, contact_infos, global_config)

        ops.update(loss_ops)
        ops['train_op'] = build_train_op(ops['loss'], ops['step'], global_config)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True, keep_checkpoint_every_n_hours=4)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        # config.log_device_placement = False
        sess = tf.Session(config=config)


        summary_ops = build_summary_ops(ops, sess, global_config)

        grasp_estimator.load_weights(sess, saver, LOG_DIR, mode='train')
        file_writers = build_file_writers(sess, LOG_DIR)

    batches_per_epoch = num_train_samples #// global_config['OPTIMIZER']['batch_size'] + 1
    cur_epoch = sess.run(ops['step']) // (batches_per_epoch * global_config['OPTIMIZER']['batch_size'])
    for epoch in range(cur_epoch, global_config['OPTIMIZER']['max_epoch']):
        log_string('**** EPOCH %03d ****' % (epoch))
        
        sess.run(ops['iterator'].initializer)
        epoch_time = time.time()
        step = train_one_epoch(sess, ops, summary_ops, file_writers, pcreader)
        print('trained %s batches in: ' % batches_per_epoch, time.time()-epoch_time)

        # Save the variables to disk.
        if (epoch+1) % 1 == 0:
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=step, write_meta_graph=False)
            log_string("Model saved in file: %s" % save_path)

        if epoch % 1 == 0 and num_test_samples > 0:
            eval_time = time.time()
            eval_test_objects(sess, ops, summary_ops, file_writers, pcreader)
            print('evaluation time: ', time.time()-eval_time)

def train_one_epoch(sess, ops, summary_ops, file_writers, pcreader):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    loss_sum, loss_sum_dir, loss_sum_ce, loss_sum_off, loss_sum_app, loss_sum_adds, loss_sum_adds_gt2pred, time_sum = 8 * [0]

    # batches_per_epoch = pcreader._num_train_samples // pcreader._batch_size
    ## define one epoch = all objects/scenes seen
    batches_per_epoch = pcreader._num_train_samples

    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, report_tensor_allocations_upon_oom = True)

    for batch_idx in range(batches_per_epoch):
        get_time = time.time()

        batch_data, cam_poses, obj_idx = pcreader.get_batch(batch_idx)
        print(time.time()- get_time)
        if 'train_on_scenes' in global_config['DATA'] and global_config['DATA']['train_on_scenes']:
            # OpenCV OpenGL conversion
            cam_poses, batch_data = center_pc_convert_cam(cam_poses, batch_data)
        print(time.time() - get_time)
        # Augment batched point clouds by rotation and jittering
        # aug_data = provider.random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25)
        if 'sigma' in global_config['DATA'] and global_config['DATA']['sigma'] > 0:
            batch_data[:,:,0:3] = provider.jitter_point_cloud(batch_data[:,:,0:3], 
                                                            sigma=global_config['DATA']['sigma'], 
                                                            clip=global_config['DATA']['clip']*2)
        
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['cam_poses_pl']: cam_poses,
                     ops['obj_idx_pl']: obj_idx,
                    #  ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}

        step, summary, _, loss_val, dir_loss, bin_ce_loss, \
        offset_loss, approach_loss, adds_loss, adds_gt2pred_loss,pos_grasps_in_view, scene_idx = sess.run([ops['step'], summary_ops['merged'], ops['train_op'], 
                                                                            ops['loss'], ops['dir_loss'], ops['bin_ce_loss'], 
                                                                            ops['offset_loss'], ops['approach_loss'], ops['adds_loss'], 
                                                                            ops['adds_gt2pred_loss'], ops['pos_grasps_in_view'], ops['scene_idx']], feed_dict=feed_dict)
        print(time.time()- get_time)
        print(pos_grasps_in_view)
        print(scene_idx, obj_idx)
        assert scene_idx[0] == obj_idx

        loss_sum += loss_val
        loss_sum_dir += dir_loss
        loss_sum_ce += bin_ce_loss
        loss_sum_off += offset_loss
        loss_sum_app += approach_loss
        loss_sum_adds += adds_loss
        loss_sum_adds_gt2pred += adds_gt2pred_loss
        time_sum += time.time() - get_time
        
        if (batch_idx+1)%10 == 0:
            file_writers['train_writer'].add_summary(summary, step)
            log_string('total loss: %f \t dir loss: %f \t ce loss: %f \t off loss: %f \t app loss: %f adds loss: %f \t adds_gt2pred loss: %f \t batch time: %f' % (loss_sum/10,loss_sum_dir/10,loss_sum_ce/10, loss_sum_off/10, loss_sum_app/10, loss_sum_adds/10, loss_sum_adds_gt2pred/10, time_sum/10))
            # log_string('accuracy: %f' % (total_correct / float(total_seen)))
            loss_sum, loss_sum_dir, loss_sum_ce, loss_sum_off, loss_sum_app, loss_sum_adds, loss_sum_adds_gt2pred, time_sum = 8 * [0]
            
    return step

def eval_test_objects(sess, ops, summary_ops, file_writers, pcreader, max_eval_objects=500):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    log_string(str(datetime.now()))
    losses = []
    losses_dir = []
    losses_ce = []
    losses_off = []
    losses_app = []
    losses_add = []
    losses_add_gt2pred = []

    # resets accumulation of pr and auc data
    sess.run(summary_ops['pr_reset_op'])
    
    for batch_idx in np.arange(min(pcreader._num_test_samples, max_eval_objects)):

        batch_data, cam_poses, obj_idx = pcreader.get_batch(obj_idx=pcreader._num_train_samples + batch_idx)

        if 'train_on_scenes' in global_config['DATA'] and global_config['DATA']['train_on_scenes']:
            # OpenCV OpenGL conversion
            cam_poses, batch_data = center_pc_convert_cam(cam_poses, batch_data)
        # Augment batched point clouds by rotation and jittering
        # aug_data = provider.random_scale_point_cloud(batch_data)
        # batch_data[:,:,0:3] = provider.jitter_point_cloud(batch_data[:,:,0:3])
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['cam_poses_pl']: cam_poses,
                     ops['obj_idx_pl']: obj_idx,
                     ops['is_training_pl']: is_training}

        scene_idx, step, loss_val, dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss, pr_summary,_,_,_ = sess.run([ops['scene_idx'], ops['step'], ops['loss'], ops['dir_loss'], ops['bin_ce_loss'],
                                                                                                        ops['offset_loss'], ops['approach_loss'], ops['adds_loss'], ops['adds_gt2pred_loss'],
                                                                                                        summary_ops['merged_eval'], summary_ops['pr_update_op'], 
                                                                                                        summary_ops['auc_update_op']] + [summary_ops['acc_update_ops']], feed_dict=feed_dict)
        assert scene_idx[0] == (pcreader._num_train_samples + batch_idx)
        losses.append(loss_val)
        losses_dir.append(dir_loss)
        losses_ce.append(bin_ce_loss)
        losses_off.append(offset_loss)
        losses_app.append(approach_loss)
        losses_add.append(adds_loss)
        losses_add_gt2pred.append(adds_gt2pred_loss)

    loss_mean = np.mean(losses)
    losses_dir_mean = np.mean(losses_dir)
    loss_ce_mean = np.mean(losses_ce)
    loss_off_mean = np.mean(losses_off)
    loss_app_mean = np.mean(losses_app)
    loss_add_mean = np.mean(losses_add)
    loss_add_gt2pred_mean = np.mean(losses_add_gt2pred)

    file_writers['test_writer'].add_summary(pr_summary, step)

    log_string('mean val loss: %f \t mean val dir loss: %f \t mean val ce loss: %f \t mean off loss: %f \t mean app loss: %f \t mean adds loss: %f \t mean adds_gt2pred loss:  %f' % (loss_mean, losses_dir_mean, loss_ce_mean, loss_off_mean, loss_app_mean, loss_add_mean, loss_add_gt2pred_mean))

    return step

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', default='/result', help='Log dir [default: log]')
    parser.add_argument('--data_path', type=str, default=None, help='internal grasp root dir')
    parser.add_argument('--max_epoch', type=int, default=None, help='Epoch to run [default: 201]')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch Size during training [default: 32]')
    parser.add_argument('--classes', nargs="*", type=str, default=None, help='train or test classes')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if not os.path.exists(FLAGS.log_dir): 
        os.makedirs(FLAGS.log_dir)

    os.system('cp pointnet2_grasp_direct.py %s' % (FLAGS.log_dir)) # bkp of model def
    os.system('cp train_grasp_direct.py %s' % (FLAGS.log_dir)) # bkp of train procedure

    LOG_FOUT = open(os.path.join(FLAGS.log_dir, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(FLAGS)+'\n')
    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)

    global_config = utilities.load_config(FLAGS.log_dir, batch_size=FLAGS.batch_size, max_epoch=FLAGS.max_epoch, data_path= FLAGS.data_path, classes=FLAGS.classes, arg_configs=FLAGS.arg_configs)
    
    log_string(str(global_config))
    log_string('pid: %s'%(str(os.getpid())))

    train(global_config, FLAGS.log_dir)

    LOG_FOUT.close()
