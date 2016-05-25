// CF JQuery Hack
require(['jquery-noconflict'], function($) {

  //Ensure MooTools is where it must be
  Window.implement('$', function(el, nc){
    return document.id(el, nc, this.document);
  });

  var $ = window.jQuery;
  //jQuery goes here


 
  // *****
  // VALIDATION
  // *****

  // browser-independent cross-domain AJAX request
  function createCrossDomainRequest(url, handler) {
    if (window.XDomainRequest) { // IE8
      return new window.XDomainRequest();
    } else {
      return new XMLHttpRequest();
    }
  }

  var request = createCrossDomainRequest();
  var serverURL = 'vystadial.ms.mff.cuni.cz:4447';
  var validationErrMsg = 'Fluency validation is not working. You will not be able to submit ' +
      'the task. Check your internet connection and/or try using a different browser (e.g., the ' +
      'latest version of Chrome).';

  // call the validation server and return the result
  function requestExternalValidation(userText, requiredData) {

    // build request URL
    var url = 'https://' + serverURL + '/?rd=' + encodeURIComponent(requiredData.join()) +
        '&ut=' + encodeURIComponent(userText);

    // send the request
    if (request) {
      // 'false' makes the call synchronous, so send() won't return before it's finished
      request.open('GET', url, false);
      request.send();
    }
    else {
      alert("Could not contact the server. " + validationErrMsg);
      return null;
    }

    // return the reply
    if (request.status == 200) {
      var response = request.responseText;
      json = JSON.parse(response);
      return json;
    }
    else {
      alert("Error while contacting the server. " + validationErrMsg);
      return null;
    }
  }

  // local validation -- just check that all data are included in the answers
  function performLocalValidation(value, data){
    return null;
    return "error msg"
  }

  function getDataItemsFor(element){
    var sys_utts = [];
    var usr_utts = [];
    var goals = [];
    var consts = [];
    $(element).closest('.html-element-wrapper').find('.history').find('.sys').find('checkemptyhistory').each(
        function(){ sys_utts.push($(this).text()); }
        );
    $(element).closest('.html-element-wrapper').find('.history').find('.user').find('checkemptyhistory').each(
        function(){ usr_utts.push($(this).text()); }
        );
    $(element).closest('.html-element-wrapper').find('.goal.checkempty').find('strong').each(
        function(){ goals.push($(this).text()); }
        );
    var role = $(element).closest('.html-element-wrapper').find('.role')[0].innerText;
    console.log('role:' + role);

    return {sys_utts: sys_utts, usr_utts: usr_utts, goals: goals, consts: consts, role: role};
  }

  // main validation method, gather data and perform local and external validation
  function validate(element) {
    var value = element.value;
    var data = getDataItemsFor(element);

    if (performLocalValidation(value, data) !== null){
      return false;
    }

    // find the corresponding hidden field
    var fluencyField = $(element).closest('.html-element-wrapper').find('.fluency_assessment')[0];
    if (fluencyField.value){  // language ID validation already performed
      var fluencyData = JSON.parse(fluencyField.value);

      if (fluencyData.result == 'yes'){
        return true;  // once the validation passes, always return true
      }
      if (fluencyData.text == value){
        return false; // never run twice for the same text
      }
    }

    return true;
    // FIXME
    // // run the external validation, return its result
    // var fluencyData = requestExternalValidation(value, data.values);
    // fluencyField.value = JSON.stringify(fluencyData);
    // return fluencyData.result == 'yes';
  }

  // return error message based on local validation
  function getErrorMessage(element){
    var value = element.value;
    var data = getDataItemsFor(element);
    var result = performLocalValidation(value, data);

    // + log all validation data (in a very crude fashion, but still)
    var validLogField = $(element).closest('.html-element-wrapper').find('.validation_log')[0];
    if (result !== null){
      validLogField.innerHTML += " /// \"" + element.value + "\" --- " + result;
      return result;
    }
    validLogField.innerHTML += " /// \"" + element.value + "\" --- not fluent";
    return 'Your reply is not fluent.';
  }

  // CrowdFlower-recommended custom validation
  // see https://success.crowdflower.com/hc/en-us/articles/201855879-Javascript-Guide-to-Customizing-a-CrowdFlower-Validator
  // This if/else block is used to hijack the functionality of an existing validator (specifically: yext_no_international_url)
  if (!_cf_cml.digging_gold) {
    CMLFormValidator.addAllThese([
      ['yext_no_international_url', {
        errorMessage: function(element) {
          return getErrorMessage(element);
        },
        validate: function(element, props) {
          // validate must return true or false
          return validate(element);
        }
      }]
    ]);
  }
  else {
    CMLFormValidator.addAllThese([
      ['yext_no_international_url', {
        validate: function(element, props) {
          return true;
        }
      }]
    ]);
  }

  // ****
  // UI
  // ****

  function filter_count_rows(cf_row_main_element) {
    var filterField = cf_row_main_element.find('.filter')[0];
    $(filterField).keyup(function () {
        var filter_val = $(this).val();
        var searches = filter_val.split(',');
        var regexps = []
        for(var i=0 ; i < searches.length ; i++){
            var rex = new RegExp(searches[i], 'i');
            regexps.push(rex)
        }

        cf_row_main_element.find('.searchable tr').hide();
        var show_count = 0;
        cf_row_main_element.find('.searchable tr').filter(function () {
            text = $(this).text();
            matches = Boolean(filter_val);
            for(var i=0; i < searches.length; i++) {
                matches = matches && regexps[i].test(text);
            }
            if (matches) {
                show_count = show_count + 1;
            }
            return matches;
        }).show();
        if (!filter_val) {
            cf_row_main_element.find('.db_instructions').show();
            console.log('hide all rows except instructions');
        } else {
            cf_row_main_element.find('.db_instructions').hide();
        }

        cf_row_main_element.find('.count').text(show_count);
        var num_rows_selected = cf_row_main_element.find('.num_rows_selected');
        num_rows_selected.value = show_count;
    });
    
  }

  function hide_non_initialized(cf_row_main_element) {
    var data = getDataItemsFor(cf_row_main_element);

        
      if (data.role == 'sys') {
        cf_row_main_element.find('.usronly').hide();
        cf_row_main_element.find('.usronly.dummyrequired').text('dummy');
        cf_row_main_element.find('.rolefull').text('hotline operator');
      } else if(data.role == 'usr') {
        cf_row_main_element.find('.sysonly').hide();
        cf_row_main_element.find('.sysonly.dummyrequired').text('dummy');
        cf_row_main_element.find('.rolefull').text('client');
      } 

      cf_row_main_element.find('.checkempty').each(function() {
        if ($(this).text() == 'No data available') {
            $(this).hide();
        }
      });

      cf_row_main_element.find('.checkemptyhistory').each(function() {
        if ($(this).text() == 'No data available') {
            $(this).parent().hide();
        }
      });
  }

  $(document).ready(function(){

    $('.html-element-wrapper').each(
        function() { 
          hide_non_initialized($(this)); 
          filter_count_rows($(this));
        }
    );
    // hide db just the dummy to show
    $('.searchable tr').hide();
    $('.db_instructions').show();

    // prevent copy-paste from the instructions
    $('.html-element-wrapper').bind("copy paste",function(e) {
      e.preventDefault(); return false;
    });

    // this will make it crash if the validation server is inaccessible
// requestExternalValidation('', []); // FIXME uncomment
  });

});
