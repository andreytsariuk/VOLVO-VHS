var express = require('express');
var router = express.Router();

router.post('/upload', function (req, res) {
  console.log(req.files.foo); // the uploaded file object
});

/* GET home page. */
router.get('/', function (req, res, next) {
  res.render('index', { title: 'Express' });
});

module.exports = router;
