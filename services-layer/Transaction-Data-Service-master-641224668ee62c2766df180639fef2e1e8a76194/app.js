var request = require("request-promise"),
  fs = require("fs"),
  name = require("node-random-name"),
  json2xls = require("json2xls"),
  vigenere = require("vigenere"),
  number = require("random-number");

/*eslint-env node*/

//------------------------------------------------------------------------------
// node.js starter application for Bluemix
//------------------------------------------------------------------------------

// This application uses express as its web server
// for more info, see: http://expressjs.com
var express = require('express');
var sex = [
  "male",
  "female"
];
// cfenv provides access to your Cloud Foundry environment
// for more info, see: https://www.npmjs.com/package/cfenv
var cfenv = require('cfenv');

// create a new express server
var app = express();

// serve the files out of ./public as our main files
app.use(express.static(__dirname + '/public'));

// get the app environment from Cloud Foundry
var appEnv = cfenv.getAppEnv();

// start server on the specified port and binding host
app.listen(appEnv.port, '0.0.0.0', function() {
  // print a message when the server starts listening
  console.log("server starting on " + appEnv.url);
});

app.get("/download", function(req, res) {
  var nopt = {
    max: 155500,
    min: 80000
  };
  var creditArray = [
    'CashOut',
    'Credit',
    'WireIn'
  ];
  var debitArray = [
    'CashIn',
    'Debit',
    'WireOut'
  ];
  var randomNumber;
  var body = [],
    transactionDetails;
  var options = {
    method: 'POST',
    url: 'https://api.us.apiconnect.ibmcloud.com/rbl/rblhackathon/rbl/v1/cas/statement',
    qs: {
      client_id: 'b6638488-c531-4ae9-b2be-6656d1ac7bff',
      client_secret: 'J2aO7pJ2xN4tQ8iW3kO1pE2nU7hA0dU5gL1iN4dV1xK1mT3mG4'
    },
    headers: {
      accept: 'application/json',
      'content-type': 'application/json'
    },
    body: {
      "Acc_Stmt_DtRng_Req": {
        "Header": {
          "TranID": "1",
          "Corp_ID": "HACKTEST",
          "Approver_ID": "A001"
        },
        "Body": {
          "Acc_No": "309002003225",
          "Tran_Type": "B",
          "From_Dt": "2010-01-01",
          "Pagination_Details": {
            "Last_Balance": {
              "Amount_Value": "",
              "Currency_Code": ""
            },
            "Last_Pstd_Date": "",
            "Last_Txn_Date": "",
            "Last_Txn_Id": "",
            "Last_Txn_SrlNo": ""
          },
          "To_Dt": "2017-08-28"
        },
        "Signature": {
          "Signature": "Signature"
        }
      }
    },
    json: true
  };

  return request(options).then((data) => {
    let memberName = [];
    transactionDetails = data.Acc_Stmt_DtRng_Res.Body.transactionDetails;
    fs.writeFile("one.json", JSON.stringify(transactionDetails[0]));
    for (let i = 0; i < transactionDetails.length; i++) {
      let member = {};
      member["source_entity_name"] = data.Acc_Stmt_DtRng_Res.Header.Corp_ID;
      member["destination_entity_name"] = vigenere.encode(name({gender: sex[i%2]}), 'avik');

      if (transactionDetails[i].transactionSummary.txnType === "C") {
        randomNumber = Math.floor(Math.random() * (creditArray.length));
        member["transfer_type"] = creditArray[randomNumber];
      } else {
        randomNumber = Math.floor(Math.random() * (debitArray.length));
        member["transfer_type"] = debitArray[randomNumber];
      }

      if (member["transfer_type"] === 'WireIn' || member["transfer_type"] === 'WireOut')
        member["account_type"] = "Foreign";
      else {
        member["account_type"] = "Domestic";
      }
      member["source_balance"] = transactionDetails[i].txnBalance.amountValue;
      member["destination_balance"] = number(nopt);
      member["transaction_date"] = transactionDetails[i].transactionSummary.txnDate;
      member["transfer_amount"] = transactionDetails[i].transactionSummary.txnAmt.amountValue;
      body.push(member);
    }
    return Promise.all(body);
  }).then((data) => {
    var xls = json2xls(data);
    fs.writeFileSync('data.xlsx', xls, 'binary');
  }).then(() => {
    return res.sendFile(__dirname + '/data.xlsx');
  }).catch((err) => {
    return res.sendFile(__dirname + '/data.xlsx');
    console.log(err);
  });
});
