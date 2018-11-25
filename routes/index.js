var express = require('express');
var router = express.Router();
const fileUpload = require('express-fileupload');
const { FilesConstroller } = require('../controllers');
router.use(fileUpload());

router.post('/upload/tooth', FilesConstroller.uploadTooth);
router.post('/upload/roller', FilesConstroller.uploadRoller);
router.post('/recognize', FilesConstroller.recognize);

/* GET home page. */
router.get('/', function (req, res, next) {
  res.render('index', { title: 'Express' });
});

module.exports = router;
