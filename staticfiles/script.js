function isNumberKeySpaceDot(evt){
    var charCode = (evt.which) ? evt.which : evt.keyCode
    if (charCode > 31 && (charCode != 32 && charCode != 46 && charCode != 45 && (charCode < 48 || charCode > 57)))
        return false;
    return true;
}

function isNumberKeyNegDot(evt){
    var charCode = (evt.which) ? evt.which : evt.keyCode
    if (charCode > 31 && (charCode != 46 && charCode != 45 && (charCode < 48 || charCode > 57)))
        return false;
    return true;
}

function isNumberKey(evt){
    var charCode = (evt.which) ? evt.which : evt.keyCode
    if (charCode < 48 || charCode > 57)
        return false;
    return true;
}

function isNumberKeyDot(evt){
    var charCode = (evt.which) ? evt.which : evt.keyCode
    if (charCode > 31 && (charCode != 46 &&(charCode < 48 || charCode > 57)))
        return false;
    return true;
}

function check(input) {
  if (parseFloat(input.value) > 1) {
    input.setCustomValidity('Your number must be between 0 and 1.');
  } else {
    input.setCustomValidity('');
  }
}